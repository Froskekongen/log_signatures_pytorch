#!/usr/bin/env python3
"""Benchmark Triton-optimized batch_sequence_tensor_product vs reference implementation.

This script compares the Triton kernel against the reference implementation
for realistic sequence lengths (100K-500K) and small batch sizes (4, 8).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, Iterable, List, Sequence

import torch

from benchmarks.utils import maybe_sync, time_call
from log_signatures_pytorch.tensor_ops import batch_sequence_tensor_product

try:
    from log_signatures_pytorch.triton_tensor_ops import (
        batch_sequence_tensor_product_triton,
    )

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    batch_sequence_tensor_product_triton = None

# Cache for compiled Triton kernels per (M, N) shape
_COMPILED_SHAPES: Dict[tuple[int, int], bool] = {}


def generate_tensor_product_inputs(
    batch: int,
    sequence: int,
    M: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random tensors for tensor product benchmark.

    Parameters
    ----------
    batch : int
        Batch size.
    sequence : int
        Sequence length.
    M : int
        First tensor dimension.
    N : int
        Second tensor dimension.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Target device.
    seed : int
        RNG seed.

    Returns
    -------
    tuple[Tensor, Tensor]
        x: (B, S, M) tensor, y: (B, S, N) tensor
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(batch, sequence, M, generator=g, dtype=dtype)
    y = torch.randn(batch, sequence, N, generator=g, dtype=dtype)
    if device.type == "cuda":
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    return x, y


def benchmark_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    measure_compile_time: bool,
    M: int,
    N: int,
) -> tuple[float, float]:
    """Benchmark Triton implementation with proper compilation handling.

    Parameters
    ----------
    x : Tensor
        Input tensor (B, S, M).
    y : Tensor
        Input tensor (B, S, N).
    device : torch.device
        CUDA device.
    warmup : int
        Warmup iterations after compilation.
    repeats : int
        Timing iterations.
    measure_compile_time : bool
        Whether to measure compilation time.
    M : int
        First tensor dimension (for cache key).
    N : int
        Second tensor dimension (for cache key).

    Returns
    -------
    tuple[float, float]
        (ms_per_call, compile_ms)
    """
    shape_key = (M, N)
    compile_ms = 0.0

    # Compile phase (first call per shape triggers autotune)
    if measure_compile_time and shape_key not in _COMPILED_SHAPES:
        maybe_sync(device)
        start_compile = time.perf_counter()
        _ = batch_sequence_tensor_product_triton(x, y)
        maybe_sync(device)
        compile_ms = (time.perf_counter() - start_compile) * 1e3
        _COMPILED_SHAPES[shape_key] = True
    elif shape_key not in _COMPILED_SHAPES:
        # Still need to compile, but don't measure time
        _ = batch_sequence_tensor_product_triton(x, y)
        maybe_sync(device)
        _COMPILED_SHAPES[shape_key] = True

    # Warmup phase
    for _ in range(warmup):
        _ = batch_sequence_tensor_product_triton(x, y)
    maybe_sync(device)

    # Timing phase
    elapsed = time_call(
        batch_sequence_tensor_product_triton,
        x,
        y,
        device=device,
        warmup=0,  # Already warmed up
        repeats=repeats,
    )
    return elapsed * 1e3, compile_ms


def benchmark_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    """Benchmark reference implementation.

    Parameters
    ----------
    x : Tensor
        Input tensor (B, S, M).
    y : Tensor
        Input tensor (B, S, N).
    device : torch.device
        Device (can be CPU or CUDA).
    warmup : int
        Warmup iterations.
    repeats : int
        Timing iterations.

    Returns
    -------
    float
        ms_per_call
    """
    elapsed = time_call(
        batch_sequence_tensor_product,
        x,
        y,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    return elapsed * 1e3


def benchmark(
    widths: Sequence[int],
    depths: Sequence[int],
    lengths: Sequence[int],
    batches: Sequence[int],
    repeats: int,
    warmup: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    measure_compile_time: bool,
) -> List[dict]:
    """Run benchmark comparing Triton vs reference implementation.

    Parameters
    ----------
    widths : Sequence[int]
        Path widths to test.
    depths : Sequence[int]
        Depths to compute M, N (M=N=width^depth).
    lengths : Sequence[int]
        Sequence lengths to test.
    batches : Sequence[int]
        Batch sizes to test.
    repeats : int
        Number of timing repeats.
    warmup : int
        Warmup iterations.
    dtype : torch.dtype
        Data type.
    device : torch.device
        Device (must be CUDA for Triton).
    seed : int
        Base RNG seed.
    measure_compile_time : bool
        Whether to measure Triton compilation time.

    Returns
    -------
    List[dict]
        List of benchmark records.
    """
    header = (
        "width depth M N length batch dtype implementation ms_per_call compile_ms speedup"
    )
    print(header)
    records: List[dict] = []

    for width in widths:
        for depth in depths:
            M = width**depth
            N = width**depth

            for length in lengths:
                for batch in batches:
                    # Generate test data
                    x, y = generate_tensor_product_inputs(
                        batch=batch,
                        sequence=length,
                        M=M,
                        N=N,
                        dtype=dtype,
                        device=device,
                        seed=seed + width + depth + length + batch,
                    )

                    try:
                        # Benchmark Triton
                        triton_ms, compile_ms = benchmark_triton(
                            x,
                            y,
                            device=device,
                            warmup=warmup,
                            repeats=repeats,
                            measure_compile_time=measure_compile_time,
                            M=M,
                            N=N,
                        )

                        record_triton = {
                            "width": width,
                            "depth": depth,
                            "M": M,
                            "N": N,
                            "length": length,
                            "batch": batch,
                            "dtype": str(dtype).replace("torch.", ""),
                            "device": device.type,
                            "implementation": "triton",
                            "ms_per_call": triton_ms,
                            "compile_ms": compile_ms,
                            "speedup": None,
                        }
                        records.append(record_triton)
                        print(
                            f"{width:>5} {depth:>5} {M:>3} {N:>3} {length:>6} {batch:>5} "
                            f"{record_triton['dtype']:>7} {'triton':>12} "
                            f"{triton_ms:10.3f} {compile_ms:10.3f} {'-':>8}"
                        )

                        # Benchmark reference (on same device)
                        ref_ms = benchmark_reference(
                            x,
                            y,
                            device=device,
                            warmup=warmup,
                            repeats=repeats,
                        )

                        record_ref = {
                            "width": width,
                            "depth": depth,
                            "M": M,
                            "N": N,
                            "length": length,
                            "batch": batch,
                            "dtype": str(dtype).replace("torch.", ""),
                            "device": device.type,
                            "implementation": "reference",
                            "ms_per_call": ref_ms,
                            "compile_ms": 0.0,
                            "speedup": None,
                        }
                        records.append(record_ref)
                        print(
                            f"{width:>5} {depth:>5} {M:>3} {N:>3} {length:>6} {batch:>5} "
                            f"{record_ref['dtype']:>7} {'reference':>12} "
                            f"{ref_ms:10.3f} {0.0:10.3f} {'-':>8}"
                        )

                        # Calculate speedup
                        speedup = ref_ms / triton_ms if triton_ms > 0 else None
                        record_triton["speedup"] = speedup
                        print(f"  -> Speedup: {speedup:.2f}x" if speedup else "  -> Speedup: N/A")

                    except (torch.cuda.OutOfMemoryError, torch.cuda.CudaError, RuntimeError) as e:
                        error_type = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "CUDA_ERROR"
                        print(
                            f"{width:>5} {depth:>5} {M:>3} {N:>3} {length:>6} {batch:>5} "
                            f"{str(dtype).replace('torch.', ''):>7} {error_type:>12} "
                            f"{'N/A':>10} {'N/A':>10} {'-':>8}"
                        )
                        # Clear cache and continue
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        try:
                            del x, y
                        except Exception:
                            pass
                        continue

                    # Clean up memory
                    del x, y
                    torch.cuda.empty_cache()

    return records


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Triton tensor product vs reference implementation."
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Path widths to benchmark.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Depths to compute M, N (M=N=width^depth).",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[100_000, 200_000, 300_000, 400_000, 500_000],
        help="Sequence lengths to benchmark.",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Timing repeats per configuration.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup runs per configuration.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Floating point precision.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--measure-compile-time",
        action="store_true",
        help="Measure one-time compile latency for Triton kernels (per shape).",
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
    return parser.parse_args(argv)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark (Triton kernels need CUDA).")

    # Check Triton availability
    if not _TRITON_AVAILABLE or batch_sequence_tensor_product_triton is None:
        raise SystemExit("Triton or batch_sequence_tensor_product_triton is not available.")

    # Set up device and dtype
    device = torch.device("cuda")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Disable gradients for benchmarking
    torch.set_grad_enabled(False)

    # Run benchmark
    try:
        records = benchmark(
            widths=args.widths,
            depths=args.depths,
            lengths=args.lengths,
            batches=args.batches,
            repeats=args.repeats,
            warmup=args.warmup,
            dtype=dtype,
            device=device,
            seed=args.seed,
            measure_compile_time=args.measure_compile_time,
        )
    except Exception as e:
        print(f"\nBenchmark encountered an error: {e}")
        print("Attempting to save partial results...")
        records = []  # Will be empty, but at least we won't crash

    # Write CSV output
    if args.output_csv and records:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = (
            args.output_csv
            if os.path.isabs(args.output_csv)
            else os.path.join(args.output_dir, args.output_csv)
        )
        fieldnames = [
            "width",
            "depth",
            "M",
            "N",
            "length",
            "batch",
            "dtype",
            "device",
            "implementation",
            "ms_per_call",
            "compile_ms",
            "speedup",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                # Convert speedup to string for CSV (None -> empty string)
                csv_record = record.copy()
                if csv_record["speedup"] is None:
                    csv_record["speedup"] = ""
                else:
                    csv_record["speedup"] = f"{csv_record['speedup']:.4f}"
                writer.writerow(csv_record)
        print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
