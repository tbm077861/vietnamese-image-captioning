"""Training loop: mixed precision, checkpoints, exact resume, logging, validation."""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from vic.config import save_yaml
from vic.engine.checkpoint import load_checkpoint, save_checkpoint
from vic.engine.data import load_experiment_data, make_image_loader, make_pair_loader
from vic.engine.evaluation import build_references, format_scores, generate_captions, score_captions
from vic.models import build_model
from vic.utils import JsonlWriter, close_logging, get_device, set_seed, setup_logging, write_json


def _make_scheduler(optimizer, kind: str, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if kind == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    return LambdaLR(optimizer, factor)


def _forward(model, batch: dict, device: torch.device, use_features: bool) -> torch.Tensor:
    visual = batch["visual"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    if use_features:
        return model(input_ids, features=visual)
    return model(input_ids, images=visual)


@torch.no_grad()
def _validation_loss(model, loader, device, use_features, use_amp, pad_id) -> float:
    """Mean cross-entropy per target token."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = _forward(model, batch, device, use_features)
        targets = batch["target_ids"].to(device)
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), targets.flatten(), ignore_index=pad_id, reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += int((targets != pad_id).sum())
    return total_loss / max(1, total_tokens)


def run_training(
    config: dict,
    resume: str | None = None,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Train according to ``config`` and return a summary of the run.

    ``resume`` is a checkpoint path, or "auto" to continue from ``last.pt`` in the run directory
    when it exists. ``max_steps`` stops early (smoke tests) without touching the schedule.
    A fresh run refuses to start in a directory that already has checkpoints unless
    ``overwrite`` is set, so a finished training is never lost by accident.
    """
    train_config = config["train"]
    run_dir = Path(config.get("output_dir", "outputs")) / config["run_name"]
    checkpoint_dir = run_dir / "checkpoints"
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    if resume is None and last_path.exists():
        if not overwrite:
            raise RuntimeError(
                f"{run_dir} already contains checkpoints. Continue with --resume auto, pick "
                "another run_name=..., or pass --overwrite to discard the previous run."
            )
        close_logging()  # an earlier run in this process may still hold train.log open
        shutil.rmtree(checkpoint_dir)
        shutil.rmtree(run_dir / "tensorboard", ignore_errors=True)
        for stale in ("metrics.jsonl", "train.log", "summary.json"):
            (run_dir / stale).unlink(missing_ok=True)
    logger = setup_logging(run_dir / "train.log")
    metrics_log = JsonlWriter(run_dir / "metrics.jsonl")

    seed = int(config.get("seed", 42))
    set_seed(seed, bool(train_config.get("deterministic", False)))
    device = get_device(config.get("device", "auto"))
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    pin_memory = device.type == "cuda"

    checkpoint = None
    if resume == "auto":
        resume = str(last_path) if last_path.exists() else None
    if resume:
        checkpoint = load_checkpoint(resume)
        logger.info(f"Resuming from {resume}")

    # --- data -------------------------------------------------------------------------------
    data = load_experiment_data(config)
    vocab = data.vocab
    if checkpoint is not None and checkpoint["vocab"] != vocab.to_list():
        raise RuntimeError("The checkpoint vocabulary differs from the prepared data")
    max_length = int(config["text"].get("max_length", 30))
    num_workers = int(train_config.get("num_workers", 0))
    train_loader, train_sampler = make_pair_loader(
        data, "train", int(train_config["batch_size"]), max_length, seed, True, num_workers,
        pin_memory,
    )  # fmt: skip
    eval_batch_size = int(config.get("eval", {}).get("batch_size", 128))
    val_loader, _ = make_pair_loader(data, "val", eval_batch_size, max_length, seed, False)
    val_image_loader = make_image_loader(data, "val", eval_batch_size)
    val_references = build_references(data.records["val"])
    steps_per_epoch = train_sampler.batches_per_epoch
    epochs = int(train_config["epochs"])

    # --- model, optimiser, schedule ---------------------------------------------------------
    model = build_model(
        config["model"], len(vocab), vocab.pad_id, with_encoder=not data.use_features
    ).to(device)
    groups = [{"params": list(model.decoder.parameters()), "lr": float(train_config["lr"])}]
    if model.encoder is not None and model.encoder.trainable:
        encoder_lr = float(train_config.get("encoder_lr", 1e-5))
        groups.append({"params": list(model.encoder.parameters()), "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train_config.get("weight_decay", 0)))
    scheduler = _make_scheduler(
        optimizer,
        train_config.get("scheduler", "cosine"),
        int(train_config.get("warmup_steps", 0)),
        epochs * steps_per_epoch,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    trainable = [p for group in groups for p in group["params"]]

    progress = {
        "epoch": 0,
        "batches_done": 0,  # batches already consumed in the current epoch
        "global_step": 0,
        "best_score": None,
        "epochs_without_improvement": 0,
    }
    if checkpoint is not None:
        model.decoder.load_state_dict(checkpoint["decoder"])
        if checkpoint.get("encoder") is not None and model.encoder is not None:
            model.encoder.load_state_dict(checkpoint["encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        progress = dict(checkpoint["progress"])
    else:
        save_yaml(config, run_dir / "config.yaml")

    writer = None
    if train_config.get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(run_dir / "tensorboard"))
        except ImportError:
            logger.info("tensorboard is not installed; skipping TensorBoard logging")

    parameter_count = sum(p.numel() for p in trainable)
    logger.info(
        f"Run {config['run_name']} | device {device} | AMP {use_amp} | "
        f"{'cached features' if data.use_features else 'images'} | vocab {len(vocab)} | "
        f"trainable params {parameter_count / 1e6:.2f}M"
    )
    logger.info(
        f"Train pairs {len(train_loader.dataset)} | val images {len(val_references)} | "
        f"{steps_per_epoch} steps/epoch | {epochs} epochs"
    )

    def save(path: Path) -> None:
        save_checkpoint(
            path, model=model, config=config, vocab=vocab, progress=progress,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        )  # fmt: skip

    log_every = int(train_config.get("log_every", 50))
    save_every = int(train_config.get("save_every_steps", 0))
    eval_every = int(train_config.get("eval_every_epochs", 1))
    patience = int(train_config.get("early_stopping_patience", 0))
    monitor = train_config.get("monitor", "CIDEr")
    grad_clip = float(train_config.get("grad_clip", 0))
    label_smoothing = float(train_config.get("label_smoothing", 0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_loss = last_loss = None
    window_loss, window_steps = 0.0, 0
    window_seconds = 0.0  # training time only; validation is timed separately
    validation_seconds = None
    timed_seconds, timed_steps = 0.0, 0
    steps_this_run = 0
    stop = False

    for epoch in range(progress["epoch"], epochs):
        if stop:
            break
        train_sampler.set_epoch(epoch, start_batch=progress["batches_done"])
        model.train()
        last_tick = time.perf_counter()
        for batch in train_loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = _forward(model, batch, device, data.use_features)
            targets = batch["target_ids"].to(device, non_blocking=True)
            loss = F.cross_entropy(
                logits.float().flatten(0, 1),
                targets.flatten(),
                ignore_index=vocab.pad_id,
                label_smoothing=label_smoothing,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {progress['global_step']}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            progress["global_step"] += 1
            progress["batches_done"] += 1
            steps_this_run += 1
            window_loss += loss.item()
            window_steps += 1
            step = progress["global_step"]
            now = time.perf_counter()
            window_seconds += now - last_tick
            last_tick = now

            if window_steps == log_every:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)  # count GPU work still in flight
                    now = time.perf_counter()
                    window_seconds += now - last_tick
                    last_tick = now
                elapsed = window_seconds
                mean_loss = window_loss / window_steps
                ms_per_step = 1000 * elapsed / window_steps
                first_loss = mean_loss if first_loss is None else first_loss
                last_loss = mean_loss
                if steps_this_run > log_every:  # skip the first window (start-up overhead)
                    timed_seconds += elapsed
                    timed_steps += window_steps
                learning_rate = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"epoch {epoch + 1} step {step} | loss {mean_loss:.4f} | "
                    f"lr {learning_rate:.2e} | {ms_per_step:.1f} ms/step"
                )
                metrics_log.write(
                    {"type": "train", "step": step, "epoch": epoch + 1, "loss": mean_loss,
                     "lr": learning_rate, "ms_per_step": ms_per_step}
                )  # fmt: skip
                if writer is not None:
                    writer.add_scalar("train/loss", mean_loss, step)
                    writer.add_scalar("train/lr", learning_rate, step)
                window_loss, window_steps = 0.0, 0
                window_seconds = 0.0

            if save_every > 0 and step % save_every == 0:
                save(last_path)
            if max_steps is not None and steps_this_run >= max_steps:
                stop = True
                break

        if stop:
            save(last_path)
            break

        # --- end of epoch: validate, checkpoint ----------------------------------------------
        validation_start = time.perf_counter()
        record = {"type": "val", "step": progress["global_step"], "epoch": epoch + 1}
        record["val_loss"] = _validation_loss(
            model, val_loader, device, data.use_features, use_amp, vocab.pad_id
        )
        message = f"epoch {epoch + 1} done | val loss {record['val_loss']:.4f}"
        improved = False
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            hypotheses = generate_captions(
                model, val_image_loader, vocab, device, data.use_features,
                beam_size=1, max_length=max_length, use_amp=use_amp,
            )  # fmt: skip
            scores = score_captions(hypotheses, val_references)
            record.update(scores)
            message += " | " + format_scores(scores)
            best = progress["best_score"]
            improved = best is None or scores[monitor] > best
            if improved:
                progress["best_score"] = scores[monitor]
                progress["epochs_without_improvement"] = 0
            else:
                progress["epochs_without_improvement"] += 1
        validation_seconds = time.perf_counter() - validation_start
        logger.info(f"{message} | validation {validation_seconds:.1f} s")
        metrics_log.write(record)
        if writer is not None:
            for key, value in record.items():
                if key not in ("type", "step", "epoch"):
                    writer.add_scalar(f"val/{key}", value, progress["global_step"])

        progress["epoch"] = epoch + 1
        progress["batches_done"] = 0
        save(last_path)
        if improved:
            save(best_path)
            logger.info(f"New best {monitor}: {progress['best_score']:.4f} -> {best_path}")
        if patience > 0 and progress["epochs_without_improvement"] >= patience:
            logger.info(f"Early stopping: no {monitor} improvement for {patience} evaluations")
            break

    if writer is not None:
        writer.close()

    ms_per_step = 1000 * timed_seconds / timed_steps if timed_steps else None
    epoch_minutes = None
    if ms_per_step is not None:
        # Training steps of one epoch plus, when measured, one validation pass.
        epoch_minutes = (ms_per_step * steps_per_epoch / 1000 + (validation_seconds or 0.0)) / 60
    summary = {
        "run_name": config["run_name"],
        "global_step": progress["global_step"],
        "steps_this_run": steps_this_run,
        "epochs_completed": progress["epoch"],
        "first_loss": first_loss,
        "last_loss": last_loss,
        "ms_per_step": ms_per_step,
        "steps_per_epoch": steps_per_epoch,
        "validation_seconds": validation_seconds,
        "estimated_epoch_minutes": epoch_minutes,
        "peak_vram_allocated_mb": None,
        "peak_vram_reserved_mb": None,
        "best_score": progress["best_score"],
    }
    if device.type == "cuda":
        summary["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        summary["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 1024**2
    write_json(summary, run_dir / "summary.json", indent=2)
    logger.info(f"Summary: {summary}")
    return summary
