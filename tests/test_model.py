"""Tests for the decoder, decoding algorithms and the image encoder wrapper."""

import pytest
import torch

from vic.models import Captioner, ImageEncoder, beam_search, build_model, greedy_decode

MODEL_CONFIG = {
    "encoder": {"name": "resnet50", "image_size": 64, "trainable": False},
    "decoder": {"type": "lstm", "embed_dim": 16, "hidden_dim": 24, "num_layers": 1, "dropout": 0.0},
}
VOCAB_SIZE, PAD, BOS, EOS = 12, 0, 1, 2


def _decoder_only():
    torch.manual_seed(0)
    return build_model(MODEL_CONFIG, VOCAB_SIZE, PAD, with_encoder=False).eval()


def test_forward_shape_and_half_precision_features():
    model = _decoder_only()
    features = torch.randn(3, 4, 2048).half()  # cached features are float16
    input_ids = torch.randint(3, VOCAB_SIZE, (3, 5))
    logits = model(input_ids, features=features)
    assert logits.shape == (3, 5, VOCAB_SIZE) and logits.dtype == torch.float32
    with pytest.raises(RuntimeError):
        model(input_ids, images=torch.zeros(3, 3, 64, 64))  # built without an encoder


def test_teacher_forcing_matches_step_by_step_decoding():
    model = _decoder_only()
    features = torch.randn(2, 4, 2048)
    input_ids = torch.randint(3, VOCAB_SIZE, (2, 6))
    full = model(input_ids, features=features)
    state = model.decoder.init_state(features)
    for position in range(input_ids.size(1)):
        logits, state = model.decoder.step(input_ids[:, position], state)
        assert torch.allclose(logits, full[:, position], atol=1e-5)


def test_greedy_and_beam_search_agree_with_beam_one_and_respect_limits():
    model = _decoder_only()
    features = torch.randn(4, 4, 2048)
    greedy = greedy_decode(model.decoder, features, BOS, EOS, max_length=7)
    assert len(greedy) == 4 and all(len(seq) <= 7 and EOS not in seq for seq in greedy)
    for index in range(4):
        beam = beam_search(
            model.decoder, features[index : index + 1], BOS, EOS, beam_size=1, max_length=7
        )
        assert beam == greedy[index]
    wide = beam_search(model.decoder, features[:1], BOS, EOS, beam_size=4, max_length=7)
    assert len(wide) <= 7 and EOS not in wide
    with pytest.raises(ValueError):
        beam_search(model.decoder, features, BOS, EOS)


def test_suppressed_tokens_are_never_generated():
    model = _decoder_only()
    features = torch.randn(3, 4, 2048)
    banned = (PAD, BOS, *range(3, VOCAB_SIZE - 1))  # leave only EOS and the last token
    for sequence in greedy_decode(model.decoder, features, BOS, EOS, 6, suppress_ids=banned):
        assert set(sequence) <= {VOCAB_SIZE - 1}
    beam = beam_search(model.decoder, features[:1], BOS, EOS, 3, 6, suppress_ids=banned)
    assert set(beam) <= {VOCAB_SIZE - 1}


def test_reorder_state_selects_batch_entries():
    model = _decoder_only()
    state = model.decoder.init_state(torch.randn(3, 4, 2048))
    hidden, cell = model.decoder.reorder_state(state, torch.tensor([2, 2, 0]))
    assert torch.equal(hidden[:, 0], state[0][:, 2]) and torch.equal(cell[:, 2], state[1][:, 0])


def test_frozen_encoder_stays_in_eval_mode_and_outputs_a_grid():
    encoder = ImageEncoder("resnet50", pretrained=False, trainable=False)
    model = Captioner(_decoder_only().decoder, encoder)
    model.train()
    assert model.decoder.training and not encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    features = model.encode(torch.zeros(2, 3, 64, 64))
    assert features.shape == (2, 4, 2048)  # 64 px input -> 2 x 2 grid
    assert not features.requires_grad
    with pytest.raises(ValueError):
        ImageEncoder("unknown")
