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

from .signature import _batch_signature, _signature_level_sizes


def knot_indices_from_repeats(
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
        Tensor of shape ``(batch,)`` with valid lengths in padded batch.
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
    >>> from log_signatures_pytorch.sparse_signature import knot_indices_from_repeats
    >>>
    >>> # Path with repeats
    >>> path = torch.tensor([
    ...     [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ... ])  # (batch=1, T=5, D=2)
    >>> knots = knot_indices_from_repeats(path)
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
        change_indices = torch.where(change_mask[b])[0] + 1  # +1 because change at t means knot at t+1
        knots = torch.cat([torch.tensor([0], device=path.device, dtype=torch.long), change_indices])

        # Ensure last valid point is included
        if lengths is not None:
            last_valid = lengths[b].item() - 1
        else:
            last_valid = seq_len - 1

        if knots[-1] != last_valid:
            knots = torch.cat([knots, torch.tensor([last_valid], device=path.device, dtype=torch.long)])

        knot_indices_list.append(knots)
        max_knots = max(max_knots, len(knots))

    # Pad to same length
    padded_knots = torch.full(
        (batch_size, max_knots), -1, dtype=torch.long, device=path.device
    )
    for b, knots in enumerate(knot_indices_list):
        padded_knots[b, : len(knots)] = knots

    return padded_knots


def sparse_increments(
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
        Tensor of shape ``(batch,)`` with valid lengths. Default is None.

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
    >>> from log_signatures_pytorch.sparse_signature import sparse_increments
    >>>
    >>> path = torch.tensor([
    ...     [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ... ])
    >>> incs, counts = sparse_increments(path)
    >>> incs.shape
    torch.Size([1, 2, 2])
    >>> counts
    tensor([3])
    """
    knots = knot_indices_from_repeats(path, eps=eps, lengths=lengths)
    batch_size, max_knots = knots.shape

    # Count valid knots per path (exclude -1 padding)
    knot_counts = (knots != -1).sum(dim=1)

    # Extract increments between consecutive knots
    max_segments = max_knots - 1
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

    # Return knot_counts (not segment_counts) as documented
    return padded_increments, knot_counts


def _unflatten_signature(sig: Tensor, width: int, depth: int) -> list[Tensor]:
    """Unflatten a signature tensor into level tensors.

    Parameters
    ----------
    sig : Tensor
        Flattened signature of shape ``(batch, sigdim)``.
    width : int
        Path width (dimension).
    depth : int
        Signature depth.

    Returns
    -------
    list[Tensor]
        List of length ``depth`` where entry ``k`` has shape
        ``(batch, width, ..., width)`` with ``k+1`` trailing width dimensions.
    """
    batch_size = sig.shape[0]
    sizes = _signature_level_sizes(width, depth)
    levels: list[Tensor] = []
    offset = 0
    for idx, size in enumerate(sizes):
        chunk = sig[:, offset : offset + size]
        shape = (batch_size,) + (width,) * (idx + 1)
        levels.append(chunk.reshape(*shape))
        offset += size
    return levels


def signature_sparse(
    path: Tensor,
    depth: int,
    eps: float = 0.0,
    lengths: Tensor | None = None,
    return_levels: bool = False,
) -> Tensor | list[Tensor]:
    """Compute sparse path signature for paths with repeated points.

    Uses Chen's identity to combine segment signatures, skipping zero
    increments (repeated points). For a path with M knots, computes the
    signature as the ordered tensor product of M-1 segment exponentials.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, T, D_aug)`` or ``(T, D_aug)`` representing
        path(s). For single path, will add batch dimension.
    depth : int
        Maximum depth L for truncation (>=1).
    eps : float, optional
        Threshold for change detection. Default is 0.0.
    lengths : Tensor, optional
        Tensor of shape ``(batch,)`` with valid lengths. Default is None.
    return_levels : bool, optional
        If True, return list of level tensors. If False, return flattened
        signature. Default is False.

    Returns
    -------
    Tensor or list[Tensor]
        If ``return_levels=False``: Tensor of shape
        ``(batch, dim + dim² + ... + dim^depth)`` with flattened signature.

        If ``return_levels=True``: List of length ``depth`` where entry ``k``
        has shape ``(batch, dim, ..., dim)`` with ``k+1`` trailing dim
        dimensions.

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
    """
    # Handle single path (add batch dimension)
    if path.ndim == 2:
        path = path.unsqueeze(0)
        single_path = True
    else:
        single_path = False

    if path.ndim != 3:
        raise ValueError(
            f"Path must be 2D (T, D) or 3D (batch, T, D); got {path.ndim}D"
        )

    if depth < 1:
        raise ValueError("depth must be >= 1")

    # Extract sparse increments (padded with zeros)
    # increments: (batch, max_segments, width)
    # The padding with zeros effectively adds identity segments (exp(0)=1),
    # which do not change the signature (S ⊗ 1 = S).
    increments, _ = sparse_increments(path, eps=eps, lengths=lengths)

    # Construct a compressed path that generates these increments.
    # We prepend a zero starting point. The absolute values don't matter for signature,
    # only the increments, so starting at 0 is fine.
    # path_compressed[t] = sum(increments[:t])
    batch_size, _, width = increments.shape
    device = path.device
    dtype = path.dtype

    zeros = torch.zeros((batch_size, 1, width), device=device, dtype=dtype)
    compressed_path_increments = torch.cat([zeros, increments], dim=1)
    compressed_path = torch.cumsum(compressed_path_increments, dim=1)
    print(f"compressed_path.shape: {compressed_path.shape}, path.shape: {path.shape}")

    # Compute signature using the vectorized implementation
    sig_flat = _batch_signature(compressed_path, depth=depth)

    if return_levels:
        return _unflatten_signature(sig_flat, width=width, depth=depth)
    else:
        if single_path:
            return sig_flat[0]
        return sig_flat
