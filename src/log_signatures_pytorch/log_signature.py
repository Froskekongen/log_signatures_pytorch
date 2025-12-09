"""Log-signature computation in Hall and Lyndon (\"words\") bases.

This module provides functions to compute the log-signature of a path, with
two coordinate systems:

- Hall basis (default): traditional Hall set ordering.
- Lyndon \"words\" basis: Signatory-style ordering where each coefficient is
  the tensor-log coefficient of a Lyndon word; projection reduces to gathers.

Both bases represent the same free Lie algebra element; a linear change of
basis relates their coordinates.
"""

from functools import lru_cache
from typing import Optional, Tuple

import torch
from torch import Tensor

from .hall_projection import logsigdim
from .lyndon_words import lyndon_words, logsigdim_words
from .hall_bch import HallBCH, supports_depth
from .hall_projection import get_hall_projector
from .signature import signature
from .tensor_ops import batch_tensor_product


@lru_cache(maxsize=None)
def _compositions(total: int, parts: int) -> Tuple[Tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    result = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            result.append((first, *rest))
    return tuple(result)


def _word_tensor_index(word: Tuple[int, ...], width: int) -> int:
    """Convert a Lyndon word to its flat tensor-algebra index (row-major)."""
    idx = 0
    for letter in word:
        idx = idx * width + (letter - 1)
    return idx


@lru_cache(maxsize=None)
def _words_indices(width: int, depth: int) -> Tuple[Tuple[int, ...], ...]:
    """Cached tensor indices for Lyndon words grouped by length."""
    grouped = [[] for _ in range(depth)]
    for word in lyndon_words(width, depth):
        grouped[len(word) - 1].append(_word_tensor_index(word, width))
    return tuple(tuple(group) for group in grouped)


def _signature_to_logsignature_tensor(
    sig_tensors: list[Tensor], width: int, depth: int
) -> list[Tensor]:
    """Convert signature tensors to log-signature tensors via log-series.

    This function implements the inverse of the exponential map in the tensor
    algebra, converting from signature coordinates to log-signature coordinates
    using the formal logarithm series.

    Parameters
    ----------
    sig_tensors : list[Tensor]
        List where entry ``k`` has shape ``(batch, width, ..., width)`` with
        ``k+1`` trailing ``width`` axes, representing the signature components
        at each depth level.
    width : int
        Path dimension (number of features).
    depth : int
        Truncation depth.

    Returns
    -------
    list[Tensor]
        List of log-signature tensors with the same shapes as ``sig_tensors``,
        where each entry represents the log-signature components at the
        corresponding depth level.

    Notes
    -----
    This is an internal function used by the default log-signature computation
    path. The conversion uses the formal logarithm series expansion.
    """
    if depth == 0 or not sig_tensors:
        return []

    device = sig_tensors[0].device
    dtype = sig_tensors[0].dtype
    batch_size = sig_tensors[0].shape[0]
    n = width
    log_sig: list[Tensor] = []

    for current_depth in range(1, depth + 1):
        if current_depth > len(sig_tensors):
            shape = [batch_size] + [n] * current_depth
            log_sig.append(torch.zeros(shape, device=device, dtype=dtype))
            continue

        accumulator = sig_tensors[current_depth - 1].clone()
        for order in range(2, current_depth + 1):
            coeff = (-1) ** (order + 1) / order
            for composition in _compositions(current_depth, order):
                term = sig_tensors[composition[0] - 1]
                for index in composition[1:]:
                    term = batch_tensor_product(term, sig_tensors[index - 1])
                accumulator = accumulator + coeff * term

        log_sig.append(accumulator)

    return log_sig


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


def _project_to_hall_basis(
    log_sig_tensors: list[Tensor], width: int, depth: int
) -> Tensor:
    """Project log-signature tensors onto Hall basis using cached projectors.

    Projects the log-signature from tensor algebra coordinates to Hall basis
    coordinates, which provides a more compact representation.

    Parameters
    ----------
    log_sig_tensors : list[Tensor]
        List of log-signature tensors in tensor algebra coordinates, where
        entry ``k`` has shape ``(batch, width, ..., width)`` with ``k+1``
        trailing width axes.
    width : int
        Path dimension (number of features).
    depth : int
        Truncation depth.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, logsigdim(width, depth))`` containing the
        log-signature in Hall basis coordinates.

    Notes
    -----
    This function uses cached projectors for efficiency. The projection matrices
    are computed once and reused for subsequent calls with the same width and depth.
    """
    if not log_sig_tensors:
        return torch.zeros(
            0,
            device=torch.device("cpu"),
            dtype=torch.float32,  # pragma: no cover
        )

    projector = get_hall_projector(
        width=width,
        depth=depth,
        device=log_sig_tensors[0].device,
        dtype=log_sig_tensors[0].dtype,
    )
    return projector.project(log_sig_tensors)


def _project_to_words_basis(
    log_sig_tensors: list[Tensor], width: int, depth: int
) -> Tensor:
    """Project log-signature tensors onto the Lyndon \"words\" basis.

    Parameters
    ----------
    log_sig_tensors : list[Tensor]
        List where entry ``k`` has shape ``(batch, width, ..., width)`` with
        ``k+1`` trailing width axes, representing log-signature tensors in
        tensor-algebra coordinates.
    width : int
        Path dimension (alphabet size).
    depth : int
        Truncation depth.

    Returns
    -------
    Tensor
        Tensor of shape ``(batch, logsigdim_words(width, depth))`` containing
        log-signature coordinates in the Lyndon \"words\" basis.

    Notes
    -----
    The Lyndon basis is triangular with respect to tensor-log coordinates, so
    each Lyndon coefficient appears exactly once; the projection is a gather
    rather than a dense matrix multiplication.
    """
    if not log_sig_tensors:
        return torch.zeros(
            0,
            device=torch.device("cpu"),
            dtype=torch.float32,  # pragma: no cover
        )

    indices_by_depth = _words_indices(width, depth)
    slices = []
    for k, indices in enumerate(indices_by_depth, start=1):
        if not indices:
            continue
        tensor = log_sig_tensors[k - 1].reshape(log_sig_tensors[k - 1].shape[0], -1)
        gather_idx = torch.tensor(indices, device=tensor.device, dtype=torch.long)
        slices.append(torch.index_select(tensor, dim=1, index=gather_idx))
    if not slices:
        # Should not happen for valid width/depth, but keep guard.
        return torch.zeros(
            log_sig_tensors[0].shape[0],
            0,
            device=log_sig_tensors[0].device,
            dtype=log_sig_tensors[0].dtype,
        )
    return torch.cat(slices, dim=1)


def _batch_log_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
    gpu_optimized: Optional[bool] = None,
    mode: str = "hall",
) -> Tensor:
    """Compute log-signatures via signature→log pipeline for batched paths.

    This implementation computes the truncated signature first, converts it to
    a tensor-log via the formal logarithm series, and then projects to either
    Hall or Lyndon (\"words\") coordinates.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
    depth : int
        Maximum depth to truncate log-signature computation.
    stream : bool, optional
        If True, computed log-signatures are returned for each step. Default is False.
    gpu_optimized : bool, optional
        Forwarded to :func:`signature`; defaults to GPU path when the input is on CUDA.
        Default is None.
    mode : str, optional
        Basis for the output coordinates: ``"hall"`` (default) or ``"words"``.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, D)`` where
        ``D = logsigdim`` for Hall or ``logsigdim_words`` for words mode.

        If ``stream=True``: Tensor of shape ``(batch, length-1, D)`` with the
        same ``D`` definition as above.

    Notes
    -----
    This is the default log-signature computation method. It works for any depth
    but may be slower than the BCH method for supported depths (depth <= 4).
    """
    mode = (mode or "hall").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")
    batch_size, seq_len, n_features = path.shape

    sig = signature(
        path,
        depth=depth,
        stream=stream,
        gpu_optimized=gpu_optimized,
    )

    projector = _project_to_hall_basis if mode == "hall" else _project_to_words_basis

    if not stream:
        sig_tensors = _unflatten_signature(sig, n_features, depth)
        log_sig_tensors = _signature_to_logsignature_tensor(
            sig_tensors, n_features, depth
        )
        return projector(log_sig_tensors, n_features, depth)

    flattened = sig.reshape(batch_size * (seq_len - 1), -1)
    sig_tensors = _unflatten_signature(flattened, n_features, depth)
    log_sig_tensors = _signature_to_logsignature_tensor(sig_tensors, n_features, depth)
    log_sig = projector(log_sig_tensors, n_features, depth)
    return log_sig.reshape(batch_size, seq_len - 1, -1)


def _batch_log_signature_bch(
    path: Tensor,
    depth: int,
    stream: bool = False,
) -> Tensor:
    """Compute log-signature via incremental BCH in Hall coordinates (depth <= 4).

    This avoids materializing the full tensor-algebra signature and
    leverages the fact that each path increment lives in the degree-1
    component of the free Lie algebra. This method is typically faster
    than the default signature→log path for supported depths.

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, width)`` representing batched paths.
    depth : int
        Truncation depth for the log-signature. Implemented exactly for
        depth <= 4; higher depths should use the default signature→log path.
    stream : bool, optional
        If True, return log-signatures at each step. Default is False.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, logsigdim(width, depth))``
        containing the final log-signature for each path.

        If ``stream=True``: Tensor of shape ``(batch, length-1, logsigdim(width, depth))``
        containing log-signatures at each step.

    Notes
    -----
    This method uses the Baker-Campbell-Hausdorff formula to incrementally
    update the log-signature. It is more memory-efficient than the default
    method but only supports depths up to 4.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.log_signature import _batch_log_signature_bch
    >>>
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> log_sig = _batch_log_signature_bch(path, depth=2)
    >>> log_sig.shape
    torch.Size([1, 3])
    """
    batch_size, seq_len, width = path.shape
    increments = torch.diff(path, dim=1)
    bch = HallBCH(width=width, depth=depth, device=path.device, dtype=path.dtype)
    steps = increments.shape[1]

    # Vectorize embedding of increments into Hall coordinates.
    hall_increments = torch.zeros(
        batch_size,
        steps,
        bch.dim,
        device=path.device,
        dtype=path.dtype,
    )
    hall_increments[:, :, :width] = increments

    state = torch.zeros(batch_size, bch.dim, device=path.device, dtype=path.dtype)
    if not stream:
        for step in range(steps):
            state = bch.bch(state, hall_increments[:, step])
        return state

    history = []
    for step in range(steps):
        state = bch.bch(state, hall_increments[:, step])
        history.append(state)
    return torch.stack(history, dim=1)


def log_signature(
    path: Tensor,
    depth: int,
    stream: bool = False,
    gpu_optimized: Optional[bool] = None,
    method: str = "default",
    mode: str = "hall",
) -> Tensor:
    """Compute log-signatures for batched paths.

    The log-signature is a compressed representation of the signature. Two bases are
    supported:

    - ``mode=\"hall\"`` (default): classic Hall basis
    - ``mode=\"words\"``: Signatory-style Lyndon words basis (triangular/gather projection)

    Parameters
    ----------
    path : Tensor
        Tensor of shape ``(batch, length, dim)`` representing batched paths.
        For a single path, pass ``path.unsqueeze(0)`` to add a batch dimension.
    depth : int
        Maximum depth to truncate log-signature computation. The output dimension
        will be ``logsigdim(dim, depth)``.
    stream : bool, optional
        If True, computed log-signatures are returned for each step. Default is False.
    gpu_optimized : bool, optional
        If True, use GPU-optimized implementation. If None, auto-detect
        (defaults to True when the input is on CUDA). Ignored for the BCH path.
        Default is None.
    method : str, optional
        Computation method: "default" (signature then log) or "bch_sparse"
        (sparse Hall-BCH, supported for depth <= 4). For higher depths,
        "bch_sparse" falls back to the default path automatically.
        Default is "default".
    mode : str, optional
        Basis for the log-signature coordinates: "hall" (default) or "words"
        (Lyndon words). "words" is only available with ``method=\"default\"``.

    Returns
    -------
    Tensor
        If ``stream=False``: Tensor of shape ``(batch, D)`` where
        ``D = logsigdim(dim, depth)`` for ``mode=\"hall\"`` and
        ``D = logsigdim_words(dim, depth)`` for ``mode=\"words\"``.

        If ``stream=True``: Tensor of shape ``(batch, length-1, D)`` with
        the same ``D`` definition as above.

    Raises
    ------
    ValueError
        If ``path`` is not three-dimensional, if ``method`` is not
        "default" or "bch_sparse", or if an unsupported ``mode``/``method``
        combination is requested.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch import log_signature, logsigdim
    >>>
    >>> # Single path (add batch dimension)
    >>> path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
    >>> log_sig = log_signature(path, depth=2)
    >>> log_sig.shape
    torch.Size([1, 3])  # logsigdim(2, 2) = 3
    >>> logsigdim(2, 2)
    3
    >>>
    >>> # Batched paths
    >>> batch_paths = torch.tensor([
    ...     [[0.0, 0.0], [1.0, 1.0]],
    ...     [[0.0, 0.0], [2.0, 2.0]],
    ... ])
    >>> log_sig = log_signature(batch_paths, depth=2)
    >>> log_sig.shape
    torch.Size([2, 3])
    >>>
    >>> # Streaming log-signatures
    >>> log_sig_stream = log_signature(path, depth=2, stream=True)
    >>> log_sig_stream.shape
    torch.Size([1, 2, 3])  # (batch, steps, logsigdim)
    >>>
    >>> # Using BCH method (faster for depth <= 4)
    >>> log_sig_bch = log_signature(path, depth=2, method="bch_sparse")
    >>> log_sig_bch.shape
    torch.Size([1, 3])
    """
    if path.ndim != 3:
        msg = (
            f"Path must be of shape (batch, path_length, path_dim); got {path.shape}. "
            "Wrap a single path with path.unsqueeze(0)."
        )
        raise ValueError(msg)

    mode = (mode or "hall").lower()
    if mode not in {"hall", "words"}:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'hall' or 'words'.")

    method = (method or "default").lower()
    if method == "bch_sparse":
        if mode != "hall":
            raise ValueError("mode='words' is only supported with method='default'.")
        if not supports_depth(depth):
            log_sig = _batch_log_signature(
                path,
                depth=depth,
                stream=stream,
                gpu_optimized=gpu_optimized,
                mode=mode,
            )
        else:
            log_sig = _batch_log_signature_bch(
                path,
                depth=depth,
                stream=stream,
            )
    elif method in {"default", None}:
        log_sig = _batch_log_signature(
            path,
            depth=depth,
            stream=stream,
            gpu_optimized=gpu_optimized,
            mode=mode,
        )
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'default' or 'bch_sparse'."
        )

    return log_sig
