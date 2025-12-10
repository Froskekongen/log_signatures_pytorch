from typing import Optional

import torch
from torch import Tensor

from .tensor_ops import (
    batch_tensor_product,
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


def _signature_level_sizes(width: int, depth: int) -> list[int]:
    return [width**k for k in range(1, depth + 1)]


def _unflatten_stream_signature(stream_sig: Tensor, width: int, depth: int) -> list[Tensor]:
    batch, steps, _ = stream_sig.shape
    sizes = _signature_level_sizes(width, depth)
    levels: list[Tensor] = []
    offset = 0
    for idx, size in enumerate(sizes):
        chunk = stream_sig[:, :, offset : offset + size]
        shape = (batch, steps) + (width,) * (idx + 1)
        levels.append(chunk.reshape(*shape))
        offset += size
    return levels


def _signature_inverse(levels: list[Tensor]) -> list[Tensor]:
    """Inverse of a truncated signature via Chen's recursion."""
    inverse: list[Tensor] = []
    for depth_index, level in enumerate(levels):
        current = -level
        for i in range(depth_index):
            current = current - batch_tensor_product(levels[i], inverse[depth_index - i - 1])
        inverse.append(current)
    return inverse


def _signature_multiply(left: list[Tensor], right: list[Tensor]) -> list[Tensor]:
    """Chen product of two truncated signatures."""
    if len(left) != len(right):
        raise ValueError("Signatures must have the same depth for multiplication.")

    product: list[Tensor] = []
    for depth_index in range(len(left)):
        current = left[depth_index] + right[depth_index]
        for i in range(depth_index):
            current = current + batch_tensor_product(left[i], right[depth_index - i - 1])
        product.append(current)
    return product


def windowed_signature(
    path: Tensor,
    depth: int,
    window_size: int,
    hop_size: int,
    gpu_optimized: Optional[bool] = None,
) -> Tensor:
    """Sliding-window signatures using Chen's identity.

    Each window signature is computed from streaming prefix signatures:
    ``Sig(path[s:e]) = Sig(path[:s])^{-1} ⊗ Sig(path[:e])`` where ``e = s + window_size - 1``.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
    depth : int
        Maximum depth to truncate signature computation.
    window_size : int
        Number of path points per window.
    hop_size : int
        Step between consecutive window starts (``>=1``).
    gpu_optimized : bool, optional
        Forwarded to :func:`signature` for CPU/GPU selection.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, dim + dim² + ... + dim^depth)``
        containing the signature of each window, flattened level-wise.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional or if windowing parameters are invalid.
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    batch_size, seq_len, width = path.shape

    if window_size < 2:
        raise ValueError("window_size must be at least 2 to form non-empty increments.")
    if hop_size < 1:
        raise ValueError("hop_size must be positive.")
    if seq_len < window_size:
        raise ValueError("window_size cannot exceed the path length.")

    num_windows = 1 + (seq_len - window_size) // hop_size

    stream = signature(path, depth=depth, stream=True, gpu_optimized=gpu_optimized)

    prefix_levels = _unflatten_stream_signature(stream, width=width, depth=depth)
    device = path.device
    dtype = path.dtype

    # Insert the identity signature at time 0 (all higher levels zero).
    for idx, level in enumerate(prefix_levels):
        zeros_shape = (batch_size, 1) + (width,) * (idx + 1)
        prefix_levels[idx] = torch.cat(
            [torch.zeros(zeros_shape, device=device, dtype=dtype), level], dim=1
        )

    start_indices = torch.arange(num_windows, device=device) * hop_size
    end_indices = start_indices + window_size - 1

    # Gather start/end signatures for all windows and merge (batch, window) into a single axis.
    start_levels: list[Tensor] = []
    end_levels: list[Tensor] = []
    for level in prefix_levels:
        start_levels.append(level[:, start_indices, ...].reshape(-1, *level.shape[2:]))
        end_levels.append(level[:, end_indices, ...].reshape(-1, *level.shape[2:]))

    inv_start = _signature_inverse(start_levels)
    window_levels = _signature_multiply(inv_start, end_levels)

    # Reshape back to (batch, num_windows, ...) and flatten level blocks.
    flattened = []
    for idx, level in enumerate(window_levels):
        reshaped = level.reshape(batch_size, num_windows, -1)
        flattened.append(reshaped)

    return torch.cat(flattened, dim=2)


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
