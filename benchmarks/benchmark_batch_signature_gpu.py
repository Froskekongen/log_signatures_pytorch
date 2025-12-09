#!/usr/bin/env python3
"""Benchmark the GPU-oriented batched signature kernel.

This script targets `_batch_signature_gpu` directly to stress the CUDA path
over a grid of widths, depths, sequence lengths, and batch sizes.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from benchmarks.utils import generate_paths, maybe_sync, time_call
from log_signatures_pytorch.log_signature import log_signature
from log_signatures_pytorch.signature import _batch_signature_gpu

_COMPILED_CACHE: Dict[
    Tuple[int, bool, str, bool, int, int, int, torch.dtype, str, str], callable
] = {}


def _make_signature_fn(
    depth: int,
    stream: bool,
    compile_mode: str,
    compile_fullgraph: bool,
    batch: int,
    length: int,
    width: int,
    dtype: torch.dtype,
    cache_compiles: bool,
    reset_dynamo: bool,
    target: str,
    mode: str,
):
    compiled = compile_mode != "none"

    if not compiled:

        def fn(path: torch.Tensor) -> torch.Tensor:
            if target == "signature":
                return _batch_signature_gpu(path, depth=depth, stream=stream)
            return log_signature(
                path,
                depth=depth,
                stream=stream,
                method="default",
                mode=mode,
            )

        return fn

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build.")

    if reset_dynamo:
        torch._dynamo.reset()

    def fn(path: torch.Tensor) -> torch.Tensor:
        if target == "signature":
            return _batch_signature_gpu(path, depth=depth, stream=stream)
        return log_signature(
            path,
            depth=depth,
            stream=stream,
            method="default",
            mode=mode,
        )

    if cache_compiles:
        key = (
            depth,
            stream,
            compile_mode,
            compile_fullgraph,
            batch,
            length,
            width,
            dtype,
            target,
            mode,
        )
        cached = _COMPILED_CACHE.get(key)
        if cached is not None:
            return cached

        compiled_fn = torch.compile(fn, mode=compile_mode, fullgraph=compile_fullgraph)
        _COMPILED_CACHE[key] = compiled_fn
        return compiled_fn

    return torch.compile(fn, mode=compile_mode, fullgraph=compile_fullgraph)


def benchmark(
    widths: Sequence[int],
    lengths: Sequence[int],
    depths: Sequence[int],
    batches: Sequence[int],
    stream: bool,
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    compile_modes: Sequence[str],
    compile_fullgraph: bool,
    cache_compiles: bool,
    reset_dynamo: bool,
    measure_compile_time: bool,
    target: str,
    mode: str,
) -> List[dict]:
    header = (
        "width depth length batch stream compiled compile_mode target basis dtype device "
        "ms_per_call compile_ms"
    )
    print(header)
    records: List[dict] = []
    for compile_mode in compile_modes:
        compiled = compile_mode != "none"
        for width in widths:
            for depth in depths:
                for length in lengths:
                    for batch in batches:
                        signature_fn = _make_signature_fn(
                            depth=depth,
                            stream=stream,
                            compile_mode=compile_mode,
                            compile_fullgraph=compile_fullgraph,
                            batch=batch,
                            length=length,
                            width=width,
                            dtype=dtype,
                            cache_compiles=cache_compiles,
                            reset_dynamo=reset_dynamo,
                            target=target,
                            mode=mode,
                        )
                        path = generate_paths(
                            batch=batch,
                            length=length,
                            width=width,
                            dtype=dtype,
                            device=device,
                            seed=seed + width + depth + length + batch,
                        )
                        compile_ms = 0.0
                        if compiled and measure_compile_time:
                            maybe_sync(device)
                            start_compile = time.perf_counter()
                            signature_fn(path)
                            maybe_sync(device)
                            compile_ms = (time.perf_counter() - start_compile) * 1e3
                        warmup_effective = warmup if not compiled else max(1, warmup)
                        elapsed = time_call(
                            signature_fn,
                            path,
                            device=device,
                            warmup=warmup_effective,
                            repeats=repeats,
                        )
                        record = {
                            "width": width,
                            "depth": depth,
                            "length": length,
                            "batch": batch,
                            "stream": stream,
                            "compiled": compiled,
                            "compile_mode": compile_mode if compiled else "none",
                            "target": target,
                            "basis": mode if target == "log_signature" else "n/a",
                            "dtype": str(dtype).replace("torch.", ""),
                            "device": device.type,
                            "ms_per_call": elapsed * 1e3,
                            "compile_ms": compile_ms,
                        }
                        records.append(record)
                        print(
                            f"{width:>5} {depth:>5} {length:>6} {batch:>5} "
                            f"{str(stream):>6} {str(compiled):>8} {record['compile_mode'][:12]:>12} "
                            f"{target[:4]:>6} {record['basis'][:5]:>6} {record['dtype']:>7} {record['device']:>6} "
                            f"{record['ms_per_call']:10.3f} {record['compile_ms']:10.3f}"
                        )
    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU batched signature kernel."
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Path widths to benchmark.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 256, 1024],
        help="Sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Signature depths to benchmark.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Benchmark streaming mode (non-streaming otherwise).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Timing repeats per configuration.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per configuration.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run on (cuda requires availability).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--compile-modes",
        type=str,
        nargs="+",
        default=["none", "reduce-overhead"],
        help="Compile modes to benchmark. Include 'none' to run uncompiled.",
    )
    parser.add_argument(
        "--no-compile-cache",
        action="store_true",
        help="Compile fresh per shape (disables compile cache; higher overhead but avoids recompile warnings).",
    )
    parser.add_argument(
        "--reset-dynamo",
        action="store_true",
        help="Call torch._dynamo.reset() before each compile (helpful when sweeping many shapes).",
    )
    parser.add_argument(
        "--measure-compile-time",
        action="store_true",
        help="Measure one-time compile latency for compiled modes (per shape).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode to use when --compiled is set.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Use fullgraph=True when compiling (default False).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to place CSV output (will be created if missing).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV filename; written under --output-dir unless an absolute path.",
    )
    parser.add_argument(
        "--target",
        choices=["signature", "log_signature"],
        default="signature",
        help="Whether to benchmark signatures or log-signatures.",
    )
    parser.add_argument(
        "--mode",
        choices=["words", "hall"],
        default="hall",
        help="Basis for log-signature target (ignored when --target=signature).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    records = benchmark(
        widths=args.widths,
        lengths=args.lengths,
        depths=args.depths,
        batches=args.batches,
        stream=args.stream,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=dtype,
        device=device,
        seed=args.seed,
        compile_modes=args.compile_modes,
        compile_fullgraph=args.compile_fullgraph,
        cache_compiles=not args.no_compile_cache,
        reset_dynamo=args.reset_dynamo,
        measure_compile_time=args.measure_compile_time,
        target=args.target,
        mode=args.mode,
    )
    if args.output_csv:
        import csv
        import os

        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = (
            args.output_csv
            if os.path.isabs(args.output_csv)
            else os.path.join(args.output_dir, args.output_csv)
        )
        fieldnames = [
            "width",
            "depth",
            "length",
            "batch",
            "stream",
            "compiled",
            "compile_mode",
            "target",
            "basis",
            "dtype",
            "device",
            "ms_per_call",
            "compile_ms",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
