"""Environment check: print PyTorch / CUDA / GPU info and run a small computation on the GPU.

Usage:
    python scripts/check_env.py

Exit code 0 means the GPU is usable; any other code means it is not.
"""

from __future__ import annotations

import platform
import time

import torch


def main() -> int:
    print(f"Python         : {platform.python_version()} ({platform.platform()})")
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA (build)   : {torch.version.cuda}")
    print(f"cuDNN          : {torch.backends.cudnn.version()}")
    try:
        import torchvision

        print(f"torchvision    : {torchvision.__version__}")
    except ImportError:
        print("torchvision    : not installed")

    if not torch.cuda.is_available():
        print("CUDA available : False")
        print("GPU check      : FAILED (a CUDA 12.8+ build of PyTorch is required)")
        return 1

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    print("CUDA available : True")
    print(f"GPU            : {props.name}")
    print(f"VRAM           : {props.total_memory / 1024**3:.1f} GB")
    print(f"Capability     : sm_{major}{minor}")
    print(f"Supported archs: {' '.join(torch.cuda.get_arch_list())}")

    torch.manual_seed(0)
    size = 4096
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    torch.cuda.synchronize()

    start = time.perf_counter()
    product = a @ b
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Compare a slice of the GPU result against the same computation on the CPU.
    reference = a[:64].cpu() @ b.cpu()
    matches_cpu = torch.allclose(product[:64].cpu(), reference, rtol=1e-3, atol=1e-2)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_product = a @ b
    amp_ok = amp_product.dtype == torch.float16 and bool(torch.isfinite(amp_product).all())

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Matmul {size}x{size}: {elapsed_ms:.1f} ms on {device}, matches CPU: {matches_cpu}")
    print(f"Mixed precision: dtype={amp_product.dtype}, finite: {amp_ok}")
    print(f"Peak GPU memory: {peak_mb:.0f} MB")

    ok = matches_cpu and amp_ok
    print(f"GPU check      : {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
