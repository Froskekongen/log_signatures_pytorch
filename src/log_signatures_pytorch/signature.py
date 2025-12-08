from typing import Optional

import torch
from torch import Tensor

from .tensor_ops import (
    batch_mult_fused_restricted_exp,
    batch_restricted_exp,
    batch_sequence_tensor_product,
)


def signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
    gpu_optimized: Optional[bool] = None,
) -> Tensor:
    """Compute signatures for batched paths.

    The signature of a path is a collection of iterated integrals that captures
    the path's geometric properties. It is computed as a truncated tensor series
    up to the specified depth.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
        For a single path, pass ``path.unsqueeze(0)`` to add a batch dimension.
    depth : int
        Maximum depth to truncate signature computation. The output dimension
        will be ``sum(dim**k for k in range(1, depth+1))``.
    stream : bool, optional
        If True, return signatures at each step along the path. If False,
        return only the final signature. Default is False.
    gpu_optimized : bool, optional
        If True, use the GPU-optimized implementation. If None, automatically
        detects based on whether the input tensor is on CUDA. Default is None.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, dim + dim² + ... + dim^depth)``
        containing the final signature for each path in the batch.

        If ``stream=True``: Tensor of shape ``(batch, length-1, dim + dim² + ... + dim^depth)``
        containing signatures at each step along each path.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch import signature
    >>>
    >>> # Single path (add batch dimension)
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> sig = signature(path, depth=2)
    >>> sig.shape
    torch.Size([1, 6])  # 2 + 4 = 6 for depth 2, width 2
    >>>
    >>> # Batched paths
    >>> batch_paths = torch.tensor([
    ...     [[0.0, 0.0], [1.0, 1.0]],
    ...     [[0.0, 0.0], [2.0, 2.0]],
    ... ])
    >>> sig = signature(batch_paths, depth=2)
    >>> sig.shape
    torch.Size([2, 6])
    >>>
    >>> # Streaming signatures
    >>> sig_stream = signature(path, depth=2, stream=True)
    >>> sig_stream.shape
    torch.Size([1, 2, 6])  # (batch, steps, sig_dim)
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    if gpu_optimized is None:
        gpu_optimized = path.is_cuda

    if gpu_optimized:
        return _batch_signature_gpu(path, depth=depth, stream=stream)
    return _batch_signature(path, depth=depth, stream=stream)


def _batch_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute signatures for batched paths on CPU using scan operations.

    This is the CPU-optimized implementation that uses sequential scan operations.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
    depth : int
        Maximum depth to truncate signature computation.
    stream : bool, optional
        If True, return signatures at each step. Default is False.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, dim + dim² + ... + dim^depth)``
        containing the final signature for each path.

        If ``stream=True``: Tensor of shape ``(batch, length-1, dim + dim² + ... + dim^depth)``
        containing signatures at each step.

    Notes
    -----
    This function is called automatically by :func:`signature` when ``gpu_optimized=False``
    or when the input tensor is on CPU.
    """
    batch_size, seq_len, n_features = path.shape
    path_increments = torch.diff(path, dim=1)  # Shape: (batch, length-1, dim)
    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)
    tail_increments = path_increments[:, 1:]

    if not stream:
        carry = exp_term
        for step in range(tail_increments.shape[1]):
            carry = batch_mult_fused_restricted_exp(tail_increments[:, step], carry)
        return torch.cat(
            [
                c.reshape(batch_size, n_features ** (1 + idx))
                for idx, c in enumerate(carry)
            ],
            dim=1,
        )
    else:
        histories = [[term] for term in exp_term]
        carry = exp_term
        for step in range(tail_increments.shape[1]):
            carry = batch_mult_fused_restricted_exp(tail_increments[:, step], carry)
            for idx, term in enumerate(carry):
                histories[idx].append(term)

        stacked = [
            torch.stack(history, dim=0)  # (steps+1, batch, ...)
            for history in histories
        ]
        return torch.cat(
            [
                torch.moveaxis(r, 1, 0).reshape(
                    batch_size, seq_len - 1, n_features ** (1 + idx)
                )
                for idx, r in enumerate(stacked)
            ],
            dim=2,
        )


def _batch_signature_gpu(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute batched signatures optimized for GPU execution.

    A memory-intensive but computationally efficient implementation that:
    - Replaces sequential scan operations with parallel matrix operations
    - Pre-computes path increment divisions
    - Uses cumulative sums for parallel sequence processing
    - Trades increased memory usage for reduced sequential operations

    Best suited when:
    - GPU VRAM can accommodate larger intermediate tensors
    - Batch/sequence sizes benefit from parallel processing
    - Computation speed is prioritized over memory efficiency

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, seq_len, features)`` representing batched paths.
    depth : int
        Maximum signature truncation depth.
    stream : bool, optional
        If True, returns signatures at each timestep. Default is False.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, features + features^2 + ... + features^depth)``
        containing the final signature for each path.

        If ``stream=True``: Tensor of shape ``(batch, seq_len-1, features + features^2 + ... + features^depth)``
        containing signatures at each timestep.

    Notes
    -----
    This function is called automatically by :func:`signature` when ``gpu_optimized=True``
    or when the input tensor is on CUDA.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.signature import _batch_signature_gpu
    >>>
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> sig = _batch_signature_gpu(path, depth=2)
    >>> sig.shape
    torch.Size([1, 6])
    """
    batch_size, seq_len, n_features = path.shape
    path_increments = torch.diff(path, dim=1)  # Shape: (batch, seq_len-1, features)

    stacked = [torch.cumsum(path_increments, dim=1)]

    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)

    path_increment_divided = None
    if depth > 1:
        path_increment_divided = torch.stack(
            [path_increments / i for i in range(2, depth + 1)], dim=0
        )

    for depth_index in range(1, depth):
        current = stacked[0][:, :-1] + path_increment_divided[depth_index - 1, :, 1:]
        for j in range(depth_index - 1):
            current = stacked[j + 1][:, :-1] + batch_sequence_tensor_product(
                current, path_increment_divided[depth_index - j - 2, :, 1:]
            )
        current = batch_sequence_tensor_product(current, path_increments[:, 1:])
        current = torch.cat([exp_term[depth_index].unsqueeze(1), current], dim=1)
        stacked.append(torch.cumsum(current, dim=1))

    if not stream:
        return torch.cat(
            [
                c[:, -1].reshape(batch_size, n_features ** (1 + idx))
                for idx, c in enumerate(stacked)
            ],
            dim=1,
        )
    else:
        return torch.cat(
            [
                r.reshape(batch_size, seq_len - 1, n_features ** (1 + idx))
                for idx, r in enumerate(stacked)
            ],
            dim=2,
        )
