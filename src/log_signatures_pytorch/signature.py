import torch
from torch import Tensor

from .tensor_ops import (
    batch_restricted_exp,
    batch_sequence_tensor_product,
    batch_tensor_product,
)


def signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
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
    torch.Size([1, 6])
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
    torch.Size([1, 2, 6])
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    return _batch_signature(path, depth=depth, stream=stream)


def _signature_level_sizes(width: int, depth: int) -> list[int]:
    return [width**k for k in range(1, depth + 1)]


def _unflatten_signature(sig: Tensor, width: int, depth: int) -> list[Tensor]:
    """Reshape flattened signature blocks into per-depth tensors.

    Converts a flattened signature tensor into a list of tensors, one for each
    depth level, where each tensor has the appropriate shape for tensor algebra
    operations.

    Parameters
    ----------
    sig : Tensor
        Flattened signature of shape ``(batch, sum(width**k for k=1..depth))``.
    width : int
        Path dimension (number of features).
    depth : int
        Truncation depth.

    Returns
    -------
    list[Tensor]
        List of length ``depth`` where entry ``k`` has shape
        ``(batch, width, ..., width)`` with ``k+1`` trailing width axes.

    Notes
    -----
    This is an internal function used to reshape signatures before converting
    to log-signatures.
    """
    if sig.ndim != 2:
        raise ValueError(
            f"Signature must be a 2D tensor of shape (batch, sigdim); got {sig.shape}."
        )

    batch = sig.shape[0]
    tensors: list[Tensor] = []
    offset = 0
    for current_depth in range(1, depth + 1):
        size = width**current_depth
        chunk = sig[:, offset : offset + size]
        shape = (batch,) + (width,) * current_depth
        tensors.append(chunk.reshape(*shape))
        offset += size
    return tensors


def _unflatten_stream_signature(
    sign_tensor: Tensor, width: int, depth: int
) -> list[Tensor]:
    """Unflatten a signature tensor into level tensors.

    This is the stream variant of :func:`_unflatten_signature` and only accepts
    stream signatures of shape ``(batch, steps, sigdim)``.

    Parameters
    ----------
    sign_tensor : Tensor
        Stream signature tensor of shape ``(batch, steps, sigdim)``.
    width : int
        Path width (dimension).
    depth : int
        Signature depth.

    Returns
    -------
    list[Tensor]
        List of length ``depth`` where entry ``k`` has shape
        ``(batch, steps, width, ..., width)`` with ``k+1`` trailing width dimensions.
    """
    if sign_tensor.ndim != 3:
        raise ValueError(
            "Stream signature must be a 3D tensor of shape (batch, steps, sigdim); "
            f"got {sign_tensor.shape}."
        )

    # Stream signature: (batch, steps, sigdim)
    batch, steps, _ = sign_tensor.shape
    # Reshape to 2D: (batch*steps, sigdim)
    flattened = sign_tensor.reshape(batch * steps, -1)
    # Unflatten to per-level tensors, then restore (batch, steps, ...) shape.
    levels = _unflatten_signature(flattened, width, depth)
    reshaped_levels: list[Tensor] = []
    for level in levels:
        level_shape = level.shape  # (batch*steps, width, ..., width)
        reshaped_levels.append(level.reshape((batch, steps) + level_shape[1:]))
    return reshaped_levels


def signature_inverse(levels: list[Tensor]) -> list[Tensor]:
    """Inverse of a truncated signature via Chen's recursion.

    Computes the inverse of a signature represented as a list of level tensors,
    where each level corresponds to a depth in the truncated signature series.
    The inverse is computed recursively using Chen's identity.

    Parameters
    ----------
    levels : list[Tensor]
        List of tensors representing the signature levels. Each tensor at index
        ``k`` should have shape ``(batch, width, ..., width)`` with ``k+1``
        trailing width dimensions, where ``width`` is the path dimension.

    Returns
    -------
    list[Tensor]
        List of tensors representing the inverse signature, with the same structure
        as the input ``levels``.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.signature import signature_inverse
    >>>
    >>> # Create a simple signature (depth=2, width=2)
    >>> level1 = torch.tensor([[[1.0, 2.0]]])  # (batch=1, width=2)
    >>> level2 = torch.tensor([[[[0.5, 0.3], [0.2, 0.1]]]])  # (batch=1, width=2, width=2)
    >>> levels = [level1, level2]
    >>>
    >>> # Compute inverse
    >>> inv_levels = signature_inverse(levels)
    >>> len(inv_levels)
    2
    """
    inverse: list[Tensor] = []
    for depth_index, level in enumerate(levels):
        current = -level
        for i in range(depth_index):
            current = current - batch_tensor_product(
                levels[i], inverse[depth_index - i - 1]
            )
        inverse.append(current)
    return inverse


def signature_multiply(left: list[Tensor], right: list[Tensor]) -> list[Tensor]:
    """Chen product of two truncated signatures.

    Computes the product of two signatures represented as lists of level tensors
    using Chen's identity. This corresponds to the signature of the concatenation
    of two paths.

    Parameters
    ----------
    left : list[Tensor]
        List of tensors representing the first signature levels. Each tensor at
        index ``k`` should have shape ``(batch, width, ..., width)`` with ``k+1``
        trailing width dimensions.
    right : list[Tensor]
        List of tensors representing the second signature levels, with the same
        structure as ``left``.

    Returns
    -------
    list[Tensor]
        List of tensors representing the product signature, with the same structure
        as the input signatures.

    Raises
    ------
    ValueError
        If ``left`` and ``right`` have different lengths (different depths).

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.signature import signature_multiply
    >>>
    >>> # Create two signatures (depth=2, width=2)
    >>> left_level1 = torch.tensor([[[1.0, 2.0]]])
    >>> left_level2 = torch.tensor([[[[0.5, 0.3], [0.2, 0.1]]]])
    >>> left = [left_level1, left_level2]
    >>>
    >>> right_level1 = torch.tensor([[[0.5, 1.0]]])
    >>> right_level2 = torch.tensor([[[[0.2, 0.1], [0.1, 0.05]]]])
    >>> right = [right_level1, right_level2]
    >>>
    >>> # Compute product
    >>> product = signature_multiply(left, right)
    >>> len(product)
    2
    """
    if len(left) != len(right):
        raise ValueError("Signatures must have the same depth for multiplication.")

    product: list[Tensor] = []
    for depth_index in range(len(left)):
        current = left[depth_index] + right[depth_index]
        for i in range(depth_index):
            current = current + batch_tensor_product(
                left[i], right[depth_index - i - 1]
            )
        product.append(current)
    return product


def _infer_width_from_signature_dim(sigdim: int, depth: int) -> int:
    """Infer path width from flattened signature dimension."""
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    width = 1
    while True:
        total = 0
        power = width
        for _ in range(depth):
            total += power
            power *= width

        if total == sigdim:
            return width

        if total > sigdim or width > sigdim:
            raise ValueError(
                f"Signature dimension {sigdim} is incompatible with depth {depth}."
            )

        width += 1


def stream_to_window_signatures(
    signature: Tensor,
    depth: int,
    window_size: int,
    hop_size: int,
) -> Tensor:
    """Compute sliding-window signatures from a stream-computed signature.

    This function applies Chen's identity to a pre-computed stream of signatures
    to obtain signatures for sliding windows.

    Parameters
    ----------
    signature : Tensor
        Tensor of shape ``(batch, length-1, dim_sum)`` containing signatures
        at each step along the path, as returned by :func:`signature(..., stream=True)`.
    depth : int
        Maximum depth of the signatures.
    window_size : int
        Number of path points per window.
    hop_size : int
        Step between consecutive window starts.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, dim_sum)`` containing the
        signature of each window.
    """
    if signature.ndim != 3:
        raise ValueError(
            "Signature must be a 3D tensor of shape "
            f"(batch, length-1, sigdim) as returned by signature(..., stream=True); "
            f"got {signature.shape}."
        )

    batch_size, stream_len, sig_dim = signature.shape
    seq_len = stream_len + 1  # Original path length

    if window_size < 2:
        raise ValueError("window_size must be at least 2 to form non-empty increments.")
    if hop_size < 1:
        raise ValueError("hop_size must be positive.")
    if seq_len < window_size:
        raise ValueError("window_size cannot exceed the path length.")

    width = _infer_width_from_signature_dim(sig_dim, depth)

    prefix_levels = _unflatten_stream_signature(signature, width=width, depth=depth)
    device = signature.device
    dtype = signature.dtype
    num_windows = 1 + (seq_len - window_size) // hop_size

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

    inv_start = signature_inverse(start_levels)
    window_levels = signature_multiply(inv_start, end_levels)

    # Reshape back to (batch, num_windows, ...) and flatten level blocks.
    flattened = []
    for idx, level in enumerate(window_levels):
        reshaped = level.reshape(batch_size, num_windows, -1)
        flattened.append(reshaped)

    return torch.cat(flattened, dim=2)


def windowed_signature(
    path: Tensor,
    depth: int,
    window_size: int,
    hop_size: int,
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

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, num_windows, dim + dim² + ... + dim^depth)``,
        where ``num_windows = 1 + (length - window_size) // hop_size``,
        containing the signature of each window, flattened level-wise.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional or if windowing parameters are invalid.

    Notes
    -----
    - Implements the Signatory-style streaming reuse (Chen) without materializing
      every window explicitly.
    - Provides the building block for :func:`windowed_log_signature`; both share
      identical window indexing and batching semantics.
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

    stream = signature(path, depth=depth, stream=True)

    return stream_to_window_signatures(stream, depth, window_size, hop_size)


def _batch_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute batched signatures using the fast parallel implementation.

    A memory-intensive but computationally efficient implementation that:
    - Replaces sequential scan operations with parallel tensor operations
    - Pre-computes path increment divisions
    - Uses cumulative sums for parallel sequence processing
    - Trades increased memory usage for reduced sequential operations

    Best suited when:
    - Memory can accommodate larger intermediate tensors
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

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.signature import _batch_signature
    >>>
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> sig = _batch_signature(path, depth=2)
    >>> sig.shape
    torch.Size([1, 6])
    """
    batch_size, seq_len, n_features = path.shape
    path_increments = torch.diff(path, dim=1)  # Shape: (batch, seq_len-1, features)

    stacked = [torch.cumsum(path_increments, dim=1)]

    exp_term = batch_restricted_exp(path_increments[:, 0], depth=depth)

    if depth > 1:
        path_increment_divided = torch.stack(
            [path_increments / i for i in range(2, depth + 1)], dim=0
        )

        for depth_index in range(1, depth):
            current = (
                stacked[0][:, :-1] + path_increment_divided[depth_index - 1, :, 1:]
            )
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
