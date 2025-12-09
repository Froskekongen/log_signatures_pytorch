#!/usr/bin/env python3
"""Benchmark default log_signature vs Hall-BCH path.

The default path computes the full signature then applies log+projection.
The BCH path works directly in Hall coordinates and should shine on long
paths / smaller depths. This script sweeps grid parameters and reports
per-call wall-clock averages and speedups.
"""

from __future__ import annotations

import argparse
import itertools
import math
from typing import List, Sequence

import torch

from benchmarks.utils import generate_paths, time_call
from log_signatures_pytorch.hall_bch import supports_depth
from log_signatures_pytorch.log_signature import log_signature


def _format_ms(value: float) -> str:
    return f"{value * 1e3:8.2f}"


def benchmark(
    lengths: Sequence[int],
    widths: Sequence[int],
    depths: Sequence[int],
    batches: Sequence[int],
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    stream: bool,
    seed: int,
    include_sparse: bool,
    mode: str,
) -> List[dict]:
    header = "width depth length batch mode stream default_ms"
    if include_sparse and mode == "hall":
        header += "  bch_sprs_ms sprs_spdup"
    print(header)
    records: List[dict] = []
    for width, depth, length, batch in itertools.product(
        widths, depths, lengths, batches
    ):
        bch_supported = supports_depth(depth)
        paths = generate_paths(
            batch=batch,
            length=length,
            width=width,
            dtype=dtype,
            device=device,
            seed=seed + length + width + depth + batch,
        )
        default_time = time_call(
            lambda p: log_signature(
                p,
                depth=depth,
                stream=stream,
                gpu_optimized=None,
                method="default",
                mode=mode,
            ),
            paths,
            device=device,
            repeats=repeats,
            warmup=warmup,
        )
        sparse_time = math.nan
        if bch_supported and include_sparse and mode == "hall":
            sparse_time = time_call(
                lambda p: log_signature(
                    p,
                    depth=depth,
                    stream=stream,
                    gpu_optimized=None,
                    method="bch_sparse",
                    mode="hall",
                ),
                paths,
                device=device,
                repeats=repeats,
                warmup=warmup,
            )
        sprs_speed = (
            default_time / sparse_time if include_sparse and bch_supported else math.nan
        )
        records.append(
            {
                "width": width,
                "depth": depth,
                "length": length,
                "batch": batch,
                "mode": mode,
                "stream": stream,
                "default_ms": default_time * 1e3,
                "bch_sparse_ms": sparse_time * 1e3,
                "sparse_speedup": sprs_speed,
            }
        )
        line = (
            f"{width:>5} {depth:>5} {length:>6} {batch:>5} "
            f"{mode:>6} {str(stream):>6} {_format_ms(default_time)}"
        )
        if include_sparse and mode == "hall":
            sprs_str = f"{_format_ms(sparse_time)}" if bch_supported else "   n/a "
            sprs_speed_str = f"{sprs_speed:7.2f}x" if bch_supported else "  n/a "
            line += f" {sprs_str} {sprs_speed_str}"
        print(line)
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark default log_signature vs Hall-BCH path."
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--repeats", type=int, default=5, help="Timing repeats per config."
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per config.")
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
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Benchmark streaming mode (non-streaming otherwise).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--include-sparse",
        action="store_true",
        help="Include scatter-based sparse BCH timings.",
    )
    parser.add_argument(
        "--mode",
        choices=["words", "hall"],
        default="hall",
        help="Basis for default log-signature path; BCH timings only support 'hall'.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path for regression tracking.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_grad_enabled(False)
    if args.include_sparse and args.mode != "hall":
        print("mode='words' selected; skipping BCH sparse timings (Hall-only).")
        args.include_sparse = False
    records = benchmark(
        lengths=args.lengths,
        widths=args.widths,
        depths=args.depths,
        batches=args.batches,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=dtype,
        device=device,
        stream=args.stream,
        seed=args.seed,
        include_sparse=args.include_sparse,
        mode=args.mode,
    )
    if args.output_csv:
        import csv

        fieldnames = [
            "width",
            "depth",
            "length",
            "batch",
            "mode",
            "stream",
            "default_ms",
            "bch_sparse_ms",
            "sparse_speedup",
        ]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
