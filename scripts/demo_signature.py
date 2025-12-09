"""Minimal demo for computing signatures or log-signatures on a toy path.

The script keeps the dependencies small and works on CPU or GPU. Use it to
verify the library is installed correctly and to see the expected shapes
returned by the chosen function.
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

import torch

from log_signatures_pytorch.signature import signature
from log_signatures_pytorch.log_signature import log_signature


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
        "--target",
        choices=["signature", "log_signature"],
        default="log_signature",
        help="Compute either the signature or the log-signature (default).",
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
    # Auto: prefer CUDA when available; otherwise use the CPU-friendly path.
    return torch.cuda.is_available()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA was requested but is not available on this machine.")

    device = torch.device(args.device)
    path = torch.randn(args.length, args.dim, device=device).unsqueeze(0)

    gpu_optimized = resolve_gpu_mode(args.gpu_optimized)
    if args.target == "signature":
        result = signature(
            path, depth=args.depth, stream=args.stream, gpu_optimized=gpu_optimized
        )
        feature_count = sum(args.dim**i for i in range(1, args.depth + 1))
    else:
        result = log_signature(
            path,
            depth=args.depth,
            stream=args.stream,
            gpu_optimized=gpu_optimized,
            mode="words",
            method="default",
        )
        from log_signatures_pytorch.lyndon_words import logsigdim_words

        feature_count = logsigdim_words(args.dim, args.depth)

    expected_shape = (
        (1, args.length - 1, feature_count) if args.stream else (1, feature_count)
    )

    print(f"Path shape (batched): {tuple(path.shape)} on {device}")
    print(f"{args.target} shape: {tuple(result.shape)} (expected {expected_shape})")
    flat = result.flatten()
    preview = flat[: min(8, flat.numel())]
    print("Preview:", preview.cpu().numpy())
    if gpu_optimized:
        print("Used GPU-optimized path implementation.")
    else:
        print("Used CPU-friendly scan implementation.")


if __name__ == "__main__":
    main()
