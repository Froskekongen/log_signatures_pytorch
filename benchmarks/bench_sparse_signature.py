"""Benchmark sparse signature vs naive signature computation."""

import time

import torch

from log_signatures_pytorch.signature import signature
from log_signatures_pytorch.sparse_signature import signature_sparse


def benchmark_sparse_vs_naive(
    batch_size: int = 1,
    path_length: int = 1000,
    width: int = 3,
    depth: int = 4,
    repeat_rate: float = 0.8,
    num_runs: int = 10,
) -> None:
    """Benchmark sparse signature vs naive signature with varying repeat rates.

    Parameters
    ----------
    batch_size : int
        Batch size for paths.
    path_length : int
        Total path length (before adding repeats).
    width : int
        Path dimension.
    depth : int
        Signature depth.
    repeat_rate : float
        Fraction of points that should be repeated (0.0 = no repeats, 1.0 = all repeats).
    num_runs : int
        Number of benchmark runs for averaging.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Create base path (random walk)
    base_increments = torch.randn(batch_size, path_length, width, device=device)
    base_path = torch.cumsum(
        torch.cat(
            [torch.zeros(batch_size, 1, width, device=device), base_increments], dim=1
        ),
        dim=1,
    )

    # Inject repeats based on repeat_rate
    path_list = []
    for b in range(batch_size):
        path_b = [base_path[b, 0]]
        for i in range(1, path_length + 1):
            path_b.append(base_path[b, i])
            # Repeat this point with probability repeat_rate
            if torch.rand(1).item() < repeat_rate:
                num_repeats = torch.randint(1, 4, (1,)).item()
                for _ in range(num_repeats):
                    path_b.append(base_path[b, i])
        path_list.append(torch.stack(path_b))

    # Pad to same length
    max_len = max(len(p) for p in path_list)
    path = torch.zeros(batch_size, max_len, width, device=device)
    for b, p in enumerate(path_list):
        path[b, : len(p)] = p

    actual_length = path.shape[1]
    print(
        f"\nBenchmark: batch_size={batch_size}, base_length={path_length}, "
        f"actual_length={actual_length}, width={width}, depth={depth}"
    )
    print(
        f"Repeat rate: {repeat_rate:.1%} (actual: {(actual_length - path_length) / actual_length:.1%})"
    )

    # Warmup
    _ = signature(path, depth=depth)
    _ = signature_sparse(path, depth=depth)

    # Benchmark naive
    times_naive = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = signature(path, depth=depth)
        times_naive.append(time.perf_counter() - start)
    avg_naive = sum(times_naive) / len(times_naive)

    # Benchmark sparse
    times_sparse = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = signature_sparse(path, depth=depth)
        times_sparse.append(time.perf_counter() - start)
    avg_sparse = sum(times_sparse) / len(times_sparse)

    speedup = avg_naive / avg_sparse
    print(
        f"Naive:    {avg_naive * 1000:.3f} ms ± {torch.std(torch.tensor(times_naive)) * 1000:.3f} ms"
    )
    print(
        f"Sparse:   {avg_sparse * 1000:.3f} ms ± {torch.std(torch.tensor(times_sparse)) * 1000:.3f} ms"
    )
    print(f"Speedup:  {speedup:.2f}x")
    print(f"Time saved: {(1 - avg_sparse / avg_naive) * 100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Sparse Signature Benchmark")
    print("=" * 60)

    # Test with low repeat rate (should be similar performance)
    benchmark_sparse_vs_naive(
        batch_size=1,
        path_length=500,
        width=3,
        depth=3,
        repeat_rate=0.1,
        num_runs=5,
    )

    # Test with medium repeat rate
    benchmark_sparse_vs_naive(
        batch_size=1,
        path_length=500,
        width=3,
        depth=3,
        repeat_rate=0.5,
        num_runs=5,
    )

    # Test with high repeat rate (should show speedup)
    benchmark_sparse_vs_naive(
        batch_size=1,
        path_length=500,
        width=3,
        depth=3,
        repeat_rate=0.9,
        num_runs=5,
    )

    # Test with larger paths and higher repeat rate
    benchmark_sparse_vs_naive(
        batch_size=1,
        path_length=10000,
        width=4,
        depth=4,
        repeat_rate=0.99,
        num_runs=5,
    )
