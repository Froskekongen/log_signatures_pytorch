import csv
import sys
import time
import tracemalloc

import numpy as np
import torch

from benchmarks.utils import maybe_sync, time_call
from log_signatures_pytorch.axis_signature import signature_axis
from log_signatures_pytorch.signature import signature
from tests.test_axis_signature import expand_axis_path


def generate_sparse_path(B, L, d, g, device="cpu"):
    """
    Generate a path where each step has exactly g changing coordinates.
    """
    path = torch.zeros(B, L, d, device=device)
    inc = torch.zeros(B, L - 1, d, device=device)

    # We can just construct random increments on CPU then move to device if needed,
    # or generate indices.
    # For speed, we just loop over steps or batch scatter.
    # Since this is setup, speed matters less than correctness.

    for b in range(B):
        for t in range(L - 1):
            indices = torch.randperm(d)[:g]
            values = torch.randn(g)
            inc[b, t, indices] = values.to(device)

    path[:, 1:] = torch.cumsum(inc, dim=1)
    return path


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Define grid
    # Reduced grid for quick run, can be expanded
    dims = [256, 512]
    lengths = [128, 512]
    batches = [8]
    gmaxs = [5, 10]
    depths = [2, 3]

    results = []

    for d in dims:
        for L in lengths:
            for B in batches:
                for g in gmaxs:
                    for depth in depths:
                        # Skip depth 3 for large d to avoid OOM or excessive wait in this demo
                        # d=256 depth=3 state size is 256^3 * 4 = 67 MB per batch item. 8 batch -> 500MB. OK.
                        # d=1024 depth=3 state size is 1GB * 4 = 4GB per batch item. 8 batch -> 32GB. SKIP.
                        if depth == 3 and d > 256:
                            continue

                        print(f"Config: B={B}, L={L}, d={d}, g={g}, depth={depth}")

                        try:
                            path = generate_sparse_path(B, L, d, g, device=device)

                            # Measure Route B (Sparse Axis)
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats()
                            tracemalloc.start()

                            t_axis = time_call(
                                signature_axis,
                                path,
                                depth,
                                False,
                                g,
                                0.0,
                                "topk",
                                "sorted",
                                False,
                                device=device,
                                warmup=2,
                                repeats=5,
                            )

                            mem_axis = 0
                            if device.type == "cuda":
                                mem_axis = torch.cuda.max_memory_allocated()
                            else:
                                _, peak = tracemalloc.get_traced_memory()
                                mem_axis = peak
                            tracemalloc.stop()

                            # Estimate throughput (steps / sec)
                            # Total steps = B * (L-1)
                            # Actually, total microsteps = B * (L-1) * g
                            # Throughput usually refers to original steps
                            throughput = (B * (L - 1)) / t_axis

                            res = {
                                "B": B,
                                "L": L,
                                "d": d,
                                "g": g,
                                "depth": depth,
                                "method": "sparse_axis",
                                "time_ms": t_axis * 1000,
                                "throughput": throughput,
                                "memory_mb": mem_axis / 1024 / 1024,
                            }
                            results.append(res)
                            print(
                                f"  Sparse: {t_axis * 1000:.2f} ms, {mem_axis / 1024 / 1024:.1f} MB"
                            )

                            # Measure Route A (Expanded Baseline)
                            # Check if feasible
                            # Expanded length L_new = 1 + (L-1)*g
                            L_new = 1 + (L - 1) * g
                            # Standard signature memory is roughly B * L_new * d^depth * 4 bytes (if fully materialized)
                            # Actually our implementation uses less, but still heavy.
                            # For d=256, depth=2: 8 * (512*10) * 65536 * 4 -> 10GB.
                            # So even d=256 is heavy for expanded baseline if L is large.
                            est_mem = B * L_new * (d**depth) * 4
                            limit = (
                                2 * 1024**3 if device.type == "cuda" else 4 * 1024**3
                            )  # 2GB GPU, 4GB CPU limit for test

                            if est_mem < limit:
                                # Expand path first (part of setup)
                                exp_path = expand_axis_path(path, g)

                                if device.type == "cuda":
                                    torch.cuda.reset_peak_memory_stats()
                                tracemalloc.start()

                                t_ref = time_call(
                                    signature,
                                    exp_path,
                                    depth,
                                    False,
                                    device=device,
                                    warmup=1,
                                    repeats=3,
                                )

                                mem_ref = 0
                                if device.type == "cuda":
                                    mem_ref = torch.cuda.max_memory_allocated()
                                else:
                                    _, peak = tracemalloc.get_traced_memory()
                                    mem_ref = peak
                                tracemalloc.stop()

                                res_ref = {
                                    "B": B,
                                    "L": L,
                                    "d": d,
                                    "g": g,
                                    "depth": depth,
                                    "method": "expanded_dense",
                                    "time_ms": t_ref * 1000,
                                    "throughput": (B * (L - 1)) / t_ref,
                                    "memory_mb": mem_ref / 1024 / 1024,
                                }
                                results.append(res_ref)
                                print(
                                    f"  Dense:  {t_ref * 1000:.2f} ms, {mem_ref / 1024 / 1024:.1f} MB"
                                )
                                print(f"  Speedup: {t_ref / t_axis:.1f}x")
                            else:
                                print(
                                    f"  Dense:  Skipped (est mem {est_mem / 1024 / 1024:.0f} MB > limit)"
                                )

                        except Exception as e:
                            print(f"  Failed: {e}")
                            import traceback

                            traceback.print_exc()

    # Write results
    keys = ["B", "L", "d", "g", "depth", "method", "time_ms", "throughput", "memory_mb"]
    with open("bench_axis_signature_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("\nBenchmark Results:")
    # Simple table print
    print(
        f"{'B':<3} {'L':<5} {'d':<5} {'g':<3} {'depth':<5} {'method':<15} {'time_ms':<10} {'mem_mb':<10}"
    )
    for r in results:
        print(
            f"{r['B']:<3} {r['L']:<5} {r['d']:<5} {r['g']:<3} {r['depth']:<5} {r['method']:<15} {r['time_ms']:<10.2f} {r['memory_mb']:<10.1f}"
        )


if __name__ == "__main__":
    run_benchmark()
