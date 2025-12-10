"""Shared benchmarking helpers for log_signatures_pytorch scripts."""

from __future__ import annotations

import time
from typing import Optional

import torch


def maybe_sync(device: torch.device) -> None:
    """Synchronize CUDA to ensure accurate timing; no-op on CPU."""

    if device.type == "cuda":
        torch.cuda.synchronize()


def generate_paths(
    batch: int,
    length: int,
    width: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
    seed: int = 0,
    cumulative: bool = True,
) -> torch.Tensor:
    """Sample random paths or increments with a fixed seed.

    Parameters
    ----------
    batch : int
        Number of paths in the batch.
    length : int
        Sequence length per path.
    width : int
        Path dimensionality.
    dtype : torch.dtype
        Desired dtype for generated tensors.
    device : torch.device, optional
        Target device for the returned tensor. Defaults to CPU when ``None``.
    seed : int, optional
        RNG seed for reproducibility.
    cumulative : bool, optional
        If True, return cumulative sums (actual paths). If False, return raw
        increments.
    """

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    increments = torch.randn(batch, length, width, generator=g, dtype=dtype)
    paths = torch.cumsum(increments, dim=1) if cumulative else increments
    if device is not None:
        paths = paths.to(device)
    return paths


def time_call(
    fn,
    *args,
    device: torch.device,
    warmup: int = 0,
    repeats: int = 1,
    synchronize: bool = True,
) -> float:
    """Time ``fn`` averaged over ``repeats`` after ``warmup`` calls."""

    for _ in range(warmup):
        fn(*args)
    if synchronize:
        maybe_sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    if synchronize:
        maybe_sync(device)
    return (time.perf_counter() - start) / repeats


def time_call_once(
    fn,
    *args,
    device: torch.device,
    synchronize: bool = True,
):
    """Run ``fn`` once and return (result, elapsed_seconds)."""

    if synchronize:
        maybe_sync(device)
    start = time.perf_counter()
    result = fn(*args)
    if synchronize:
        maybe_sync(device)
    return result, time.perf_counter() - start
