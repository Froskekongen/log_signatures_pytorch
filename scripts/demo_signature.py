"""Minimal demo for computing log signatures on a toy path.

The script keeps the dependencies small and works on CPU or GPU. Use it to
verify the library is installed correctly and to see the expected shapes
returned by the ``signature`` function.
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

import torch

from log_signatures_pytorch.signature import signature


GpuMode = Literal["auto", "on", "off"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a log signature for a random path."
    )
    parser.add_argument(
        "--length", type=int, default=8, help="Number of points in the path."
    )
    parser.add_argument(
        "--dim", type=int, default=3, help="Dimensionality of each point."
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Depth to truncate the signature."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Return the signature at every step instead of only the final value.",
    )
    parser.add_argument(
        "--gpu-optimized",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use the GPU-optimized parallel implementation ('auto' picks CUDA when available).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (e.g. 'cpu', 'cuda', or 'cuda:0').",
    )
    return parser.parse_args()


def resolve_gpu_mode(mode: GpuMode) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    # Default to the GPU-optimized path because it is also the preferred code
    # path for CPU execution in this repository.
    return True


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA was requested but is not available on this machine.")

    device = torch.device(args.device)
    path = torch.randn(args.length, args.dim, device=device)

    gpu_optimized = resolve_gpu_mode(args.gpu_optimized)
    sig = signature(
        path, depth=args.depth, stream=args.stream, gpu_optimized=gpu_optimized
    )

    feature_count = sum(args.dim**i for i in range(1, args.depth + 1))
    expected_shape = (
        (args.length - 1, feature_count) if args.stream else (feature_count,)
    )

    print(f"Path shape: {tuple(path.shape)} on {device}")
    print(f"Signature shape: {tuple(sig.shape)} (expected {expected_shape})")
    flat = sig.flatten()
    preview = flat[: min(8, flat.numel())]
    print("Preview:", preview.cpu().numpy())
    if gpu_optimized:
        print("Used GPU-optimized path implementation.")
    else:
        print("Used CPU-friendly scan implementation.")


if __name__ == "__main__":
    main()
