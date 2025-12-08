#!/usr/bin/env python3
"""Benchmark fused restricted exponential updates (batched paths only)."""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Sequence

import torch

from log_signatures_pytorch.tensor_ops import (
    batch_mult_fused_restricted_exp,
    batch_restricted_exp,
)


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _generate_batch_paths(
    batch_size: int,
    length: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    increments = torch.randn(batch_size, length, width, generator=g, dtype=dtype)
    return increments.to(device)


def _run_batch(path_increments: torch.Tensor, depth: int) -> List[torch.Tensor]:
    carry = batch_restricted_exp(path_increments[:, 0], depth=depth)
    for step in range(1, path_increments.shape[1]):
        carry = batch_mult_fused_restricted_exp(path_increments[:, step], carry)
    return carry


def _time_call(fn, *args, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn(*args)
    _maybe_sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    _maybe_sync(device)
    return (time.perf_counter() - start) / repeats


def benchmark(
    lengths: Sequence[int],
    width: int,
    depth: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> list[dict]:
    results: list[dict] = []
    header = "length batch width depth dtype device ms_per_call"
    print(header)
    for length in lengths:
        path = _generate_batch_paths(
            batch_size=batch_size,
            length=length,
            width=width,
            dtype=dtype,
            device=device,
            seed=seed + length,
        )
        fn = _run_batch
        args = (path, depth)
        elapsed = _time_call(
            fn,
            *args,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        record = {
            "length": length,
            "batch": batch_size,
            "width": width,
            "depth": depth,
            "dtype": str(dtype).replace("torch.", ""),
            "device": device.type,
            "ms_per_call": elapsed * 1e3,
        }
        results.append(record)
        print(
            f"{length:>6} {batch_size:>5} {width:>5} {depth:>5} "
            f"{record['dtype']:>7} {record['device']:>6} {record['ms_per_call']:10.3f}"
        )
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused restricted exponential updates."
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 256, 1024],
        help="Sequence lengths to benchmark.",
    )
    parser.add_argument("--width", type=int, default=3, help="Path width.")
    parser.add_argument("--depth", type=int, default=4, help="Signature depth.")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for batched mode."
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Timing repeats per configuration."
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup runs per configuration."
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on (cuda requires availability).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path for regression tracking.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    results = benchmark(
        lengths=args.lengths,
        width=args.width,
        depth=args.depth,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    if args.output_csv:
        import csv

        fieldnames = [
            "length",
            "batch",
            "width",
            "depth",
            "dtype",
            "device",
            "ms_per_call",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
