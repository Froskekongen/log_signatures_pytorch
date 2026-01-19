"""Sparse path signature computation for paths with repeated points.

This module implements efficient signature computation for augmented paths that
contain many repeated points (sparse change points). The key insight is that
repeated points imply zero increments, whose segment signature is the identity,
so they can be skipped.

The implementation uses Chen's identity to combine segment signatures:
- For a linear segment with displacement v, the truncated segment signature
  is E_L(v) = sum_{k=0..L} v^{⊗k}/k!
- The full signature is the ordered tensor product of segment exponentials
"""

import torch
from torch import Tensor

from .signature import (
    _batch_signature,
    _unflatten_signature,
    _unflatten_stream_signature,
)


def pad_paths_correctly(
    paths: list[Tensor], max_length: int | None = None
) -> tuple[Tensor, Tensor]:
    """Pad variable-length paths by repeating each path's last point.

    This is the recommended padding strategy when batching variable-length paths
    for signatures/log-signatures: repeating the final valid point produces zero
    increments on the padded tail, so the signature remains unchanged.

    When using this padding strategy, you typically do **not** need to pass a
    ``lengths`` tensor to :func:`signature_sparse`/sparse log-signature calls,
    because the padding does not affect the result.

    Parameters
    ----------
    paths : list[Tensor]
        List of tensors shaped ``(T_i, D)`` with possibly different lengths
        ``T_i``. Each path must have at least one point.
    max_length : int, optional
        Target padded length. If None, uses ``max(T_i)``.

    Returns
    -------
    tuple[Tensor, Tensor]
        - padded_paths: Tensor of shape ``(batch, max_length, D)``
        - lengths: Tensor of shape ``(batch,)`` containing the original lengths

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.sparse_signature import pad_paths_correctly, signature_sparse
    >>>
    >>> paths = [
    ...     torch.tensor([[0.0], [1.0], [2.0]]),  # length 3
    ...     torch.tensor([[0.0], [1.0]]),         # length 2
    ... ]
    >>> padded, lengths = pad_paths_correctly(paths)
    >>> padded.shape
    torch.Size([2, 3, 1])
    >>> lengths
    tensor([3, 2])
    >>> # Padding is "signature-safe" (zero increments), so lengths is optional here:
    >>> sig = signature_sparse(padded, depth=2)
    """
    if len(paths) == 0:
        raise ValueError("paths must be a non-empty list of tensors")

    lengths_list = [int(p.shape[0]) for p in paths]
    if any(length <= 0 for length in lengths_list):
        raise ValueError("each path must have at least one point (T_i >= 1)")

    first = paths[0]
    if first.ndim != 2:
        raise ValueError(
            f"each path must have shape (T, D); got {first.shape} for paths[0]"
        )
    d = int(first.shape[1])
    device = first.device
    dtype = first.dtype

    for i, p in enumerate(paths[1:], start=1):
        if p.ndim != 2:
            raise ValueError(f"each path must have shape (T, D); got {p.shape} at {i}")
        if int(p.shape[1]) != d:
            raise ValueError(
                f"all paths must have the same D; got {d} and {int(p.shape[1])} at {i}"
            )
        if p.device != device:
            raise ValueError("all paths must be on the same device")
        if p.dtype != dtype:
            raise ValueError("all paths must have the same dtype")

    if max_length is None:
        max_length = max(lengths_list)
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be >= 1")
    if max_length < max(lengths_list):
        raise ValueError("max_length must be >= max(path lengths)")

    batch = len(paths)
    padded = torch.empty((batch, max_length, d), device=device, dtype=dtype)
    lengths = torch.tensor(lengths_list, device=device, dtype=torch.long)

    for b, p in enumerate(paths):
        t = int(p.shape[0])
        padded[b, :t] = p
        if t < max_length:
            padded[b, t:] = p[-1].expand(max_length - t, -1)

    return padded, lengths


def _knot_indices_from_repeats(
    path: Tensor, eps: float = 0.0, lengths: Tensor | None = None
) -> Tensor:
    """Extract knot indices where the path changes (non-repeated points).

    For each consecutive pair, computes the change and marks indices where
    the path changes by more than eps. Always includes index 0, and includes
    the last valid point for each path in a batch.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, T, D_aug)`` representing batched paths.
    eps : float, optional
        Threshold for change detection. If ``eps==0``, exact comparison.
        Otherwise, a point is marked as "changed" if ``max(abs(delta)) > eps``.
        Default is 0.0.
    lengths : Tensor, optional
        Tensor of shape ``(batch,)`` with valid lengths in a padded batch.

        **Padding guideline**
        - If you pad by repeating the last valid point (signature-safe padding),
          you can usually leave this as None.
        - If you pad with zeros/any other values, pass ``lengths`` to ignore the
          padded tail (otherwise padding can introduce spurious changes).

        If None, all T points are considered valid. Default is None.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, M)`` where M is the number of knots per path
        (may vary). Each row contains the indices where changes occur, always
        starting with 0. Padded with -1 for paths with fewer knots.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.sparse_signature import _knot_indices_from_repeats
    >>>
    >>> # Path with repeats
    >>> path = torch.tensor([
    ...     [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ... ])  # (batch=1, T=5, D=2)
    >>> knots = _knot_indices_from_repeats(path)
    >>> knots
    tensor([[0, 2, 4]])
    """
    batch_size, seq_len, n_features = path.shape

    if seq_len < 2:
        # Single point or empty path - return just index 0
        return torch.zeros((batch_size, 1), dtype=torch.long, device=path.device)

    # Compute increments: (batch, seq_len-1, n_features)
    increments = torch.diff(path, dim=1)

    # Change detection: max(abs(delta)) > eps
    if eps == 0.0:
        # Exact comparison: any nonzero increment
        change_mask = increments.abs().amax(dim=2) > 0.0
    else:
        # Threshold-based: max(abs(delta)) > eps
        change_mask = increments.abs().amax(dim=2) > eps

    # Handle padded batches
    if lengths is not None:
        # Only consider changes within valid length
        valid_mask = torch.arange(seq_len - 1, device=path.device).unsqueeze(0) < (
            lengths.unsqueeze(1) - 1
        )
        change_mask = change_mask & valid_mask

    # Always include index 0
    knot_indices_list = []
    max_knots = 0

    for b in range(batch_size):
        # Find indices where change occurs (these are the start of new segments)
        # The knot at index t+1 corresponds to change detected at increment t
        change_indices = (
            torch.where(change_mask[b])[0] + 1
        )  # +1 because change at t means knot at t+1
        knots = torch.cat(
            [torch.tensor([0], device=path.device, dtype=torch.long), change_indices]
        )

        # Ensure last valid point is included
        if lengths is not None:
            last_valid = lengths[b].item() - 1
        else:
            last_valid = seq_len - 1

        if knots[-1] != last_valid:
            knots = torch.cat(
                [
                    knots,
                    torch.tensor([last_valid], device=path.device, dtype=torch.long),
                ]
            )

        knot_indices_list.append(knots)
        max_knots = max(max_knots, len(knots))

    # Pad to same length
    padded_knots = torch.full(
        (batch_size, max_knots), -1, dtype=torch.long, device=path.device
    )
    for b, knots in enumerate(knot_indices_list):
        padded_knots[b, : len(knots)] = knots

    return padded_knots


def _sparse_increments_and_knots(
    path: Tensor, eps: float = 0.0, lengths: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract sparse increments and knot indices.

    Internal helper for _sparse_increments and signature_sparse.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        - increments: (batch, M-1, D)
        - knot_counts: (batch,)
        - knots: (batch, M)
    """
    knots = _knot_indices_from_repeats(path, eps=eps, lengths=lengths)
    batch_size, max_knots = knots.shape

    # Count valid knots per path (exclude -1 padding)
    knot_counts = (knots != -1).sum(dim=1)

    # Extract increments between consecutive knots
    increments_list = []

    for b in range(batch_size):
        num_knots = knot_counts[b].item()
        if num_knots < 2:
            # No segments (single point or empty)
            increments_list.append(
                torch.zeros((0, path.shape[2]), device=path.device, dtype=path.dtype)
            )
            continue

        path_knots = knots[b, :num_knots]
        # Get path values at knot indices
        knot_values = path[b, path_knots]  # (num_knots, D_aug)
        # Compute increments between consecutive knots
        path_increments = torch.diff(knot_values, dim=0)  # (num_knots-1, D_aug)
        increments_list.append(path_increments)

    # Pad to same length
    max_segments = max(len(inc) for inc in increments_list) if increments_list else 0
    if max_segments == 0:
        max_segments = 1  # At least one dimension for empty case

    padded_increments = torch.zeros(
        (batch_size, max_segments, path.shape[2]),
        device=path.device,
        dtype=path.dtype,
    )
    segment_counts = torch.zeros((batch_size,), dtype=torch.long, device=path.device)

    for b, inc in enumerate(increments_list):
        num_segments = len(inc)
        if num_segments > 0:
            padded_increments[b, :num_segments] = inc
        segment_counts[b] = num_segments

    return padded_increments, knot_counts, knots


def _sparse_increments(
    path: Tensor, eps: float = 0.0, lengths: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Extract sparse (non-zero) increments between knots.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, T, D_aug)`` representing batched paths.
    eps : float, optional
        Threshold for change detection. Default is 0.0.
    lengths : Tensor, optional
        Tensor of shape ``(batch,)`` with valid lengths in a padded batch.

        See :func:`pad_paths_correctly` for the recommended padding strategy.
        If you pad with zeros/any other values, pass ``lengths`` to ignore the
        padded tail.

        Default is None.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple of:
        - increments: Tensor of shape ``(batch, M-1, D_aug)`` where M is the
          number of knots. Contains the displacement between consecutive knots.
        - knot_counts: Tensor of shape ``(batch,)`` with the number of knots
          per path.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.sparse_signature import _sparse_increments
    >>>
    >>> path = torch.tensor([
    ...     [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ... ])
    >>> incs, counts = _sparse_increments(path)
    >>> incs.shape
    torch.Size([1, 2, 2])
    >>> counts
    tensor([3])
    """
    increments, knot_counts, _ = _sparse_increments_and_knots(
        path, eps=eps, lengths=lengths
    )
    return increments, knot_counts


def _expand_stream_signature(
    compressed_sig: Tensor,
    knots: Tensor,
    total_steps: int,
    knot_counts: Tensor,
) -> Tensor:
    """Expand compressed stream signature to full path length.

    Propagates signatures forward from knots to subsequent repeated points.
    """
    batch_size, _, sig_dim = compressed_sig.shape
    device = compressed_sig.device
    dtype = compressed_sig.dtype

    # Create output tensor (batch, total_steps, sig_dim)
    # total_steps is usually seq_len - 1
    full_sig = torch.zeros(
        (batch_size, total_steps, sig_dim), device=device, dtype=dtype
    )

    # Prepare compressed signatures with identity prepended
    # compressed_sig has M-1 steps. We add step 0 (identity/zeros).
    # Shape: (batch, M, sig_dim)
    zeros = torch.zeros((batch_size, 1, sig_dim), device=device, dtype=dtype)
    sig_padded = torch.cat([zeros, compressed_sig], dim=1)

    # Create mapping indices
    # We want to map each t in 1..total_steps to index j in sig_padded
    # such that knots[j] <= t < knots[j+1].
    # This corresponds to torch.searchsorted(knots, t, side='right') - 1.

    # Fix knots padding: replace -1 with infinity (or > total_steps)
    # so searchsorted works correctly.
    # knots is (batch, max_knots)
    knots_fixed = knots.clone()
    # Mask for padding (-1)
    mask = knots_fixed == -1
    # Replace -1 with a large number
    knots_fixed[mask] = total_steps + 2

    # Time indices to query: 1, 2, ..., total_steps
    t_values = (
        torch.arange(1, total_steps + 1, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    # Find insertion points
    # (batch, total_steps)
    indices = torch.searchsorted(knots_fixed, t_values, side="right") - 1

    # Clamp indices to be safe (should be >= 0)
    indices = indices.clamp(min=0)

    # Gather values
    # We need to gather from sig_padded along dim 1
    # sig_padded: (batch, M, sig_dim)
    # indices: (batch, total_steps)
    # Expand indices to (batch, total_steps, sig_dim)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, sig_dim)

    # Gather
    full_sig = torch.gather(sig_padded, 1, indices_expanded)

    return full_sig


def signature_sparse(
    path: Tensor,
    depth: int,
    eps: float = 0.0,
    lengths: Tensor | None = None,
    return_levels: bool = False,
    stream: bool = False,
) -> Tensor | list[Tensor]:
    """Compute sparse path signature for paths with repeated points.

    Uses Chen's identity to combine segment signatures, skipping zero
    increments (repeated points). For a path with M knots, computes the
    signature as the ordered tensor product of M-1 segment exponentials.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, T, D_aug)`` representing batched paths.
        For a single path, pass ``path.unsqueeze(0)`` to add a batch dimension.
    depth : int
        Maximum depth L for truncation (>=1).
    eps : float, optional
        Threshold for change detection. Default is 0.0.
    lengths : Tensor, optional
        Tensor of shape ``(batch,)`` with valid lengths in a padded batch.

        **Best practice (recommended)**: pad by repeating the last valid point
        of each path (see :func:`pad_paths_correctly`). This padding produces
        zero increments on the tail, so it does not change the signature and
        you can usually leave ``lengths=None``.

        If you instead pad with zeros/any other values, you must pass ``lengths``
        to ignore the padded tail (otherwise padding can introduce spurious
        increments and change the result).

        Default is None.
    return_levels : bool, optional
        If True, return list of level tensors. If False, return flattened
        signature. Default is False.
    stream : bool, optional
        If True, return signatures at each step along the path. If False,
        return only the final signature. Default is False.

    Returns
    -------
    Tensor or list[Tensor]
        If ``return_levels=False``: Tensor of shape
        ``(batch, dim_sig)`` or ``(batch, T-1, dim_sig)`` (if stream=True).

        If ``return_levels=True``: List of tensors, either final signatures
        or streams depending on ``stream`` argument.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.sparse_signature import signature_sparse
    >>>
    >>> # Path with repeats
    >>> path = torch.tensor([
    ...     [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ... ])
    >>> sig = signature_sparse(path, depth=2)
    >>> sig.shape
    torch.Size([1, 6])
    >>> stream_sig = signature_sparse(path, depth=2, stream=True)
    >>> stream_sig.shape
    torch.Size([1, 4, 6])
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, T, D_aug); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    if depth < 1:
        raise ValueError("depth must be >= 1")

    # Extract sparse increments (padded with zeros)
    # increments: (batch, max_segments, width)
    increments, knot_counts, knots = _sparse_increments_and_knots(
        path, eps=eps, lengths=lengths
    )

    # Construct a compressed path that generates these increments.
    # We prepend a zero starting point.
    batch_size, _, width = increments.shape
    device = path.device
    dtype = path.dtype

    zeros = torch.zeros((batch_size, 1, width), device=device, dtype=dtype)
    compressed_path_increments = torch.cat([zeros, increments], dim=1)
    compressed_path = torch.cumsum(compressed_path_increments, dim=1)

    # Compute signature using the vectorized implementation
    # If stream=True, we need the streaming signature of the compressed path
    # to reconstruct the full stream.
    sig_result = _batch_signature(compressed_path, depth=depth, stream=stream)

    if stream:
        # Map compressed stream back to full path
        seq_len = path.shape[1]

        # sig_result is (batch, max_segments, sig_dim)
        # We need to expand to (batch, seq_len-1, sig_dim)
        sig_result = _expand_stream_signature(
            sig_result, knots, seq_len - 1, knot_counts
        )

    if return_levels:
        if stream:
            return _unflatten_stream_signature(sig_result, width=width, depth=depth)
        return _unflatten_signature(sig_result, width=width, depth=depth)
    else:
        return sig_result
