from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Union

import torch

from .basis import HallBasisElement, hall_basis
from .tensor_ops import lie_brackets


@lru_cache(maxsize=None)
def _hall_element_depth(elem: HallBasisElement) -> int:
    if isinstance(elem, int):
        return 1
    left, right = elem
    return _hall_element_depth(left) + _hall_element_depth(right)


@lru_cache(maxsize=None)
def _hall_basis_with_depths(
    width: int, depth: int
) -> Tuple[Tuple[HallBasisElement, ...], Dict[int, Tuple[HallBasisElement, ...]]]:
    basis_tuple = tuple(hall_basis(width, depth))
    depth_map: Dict[int, List[HallBasisElement]] = {}
    for elem in basis_tuple:
        elem_depth = _hall_element_depth(elem)
        depth_map.setdefault(elem_depth, []).append(elem)
    return basis_tuple, {k: tuple(v) for k, v in depth_map.items()}


@lru_cache(maxsize=None)
def _hall_basis_tensors(width: int, depth: int) -> Dict[HallBasisElement, torch.Tensor]:
    basis, _ = _hall_basis_with_depths(width, depth)
    cache: Dict[HallBasisElement, torch.Tensor] = {}

    def build(elem: HallBasisElement) -> torch.Tensor:
        if elem in cache:
            return cache[elem]
        if isinstance(elem, int):
            tensor = torch.zeros(width, dtype=torch.float64)
            tensor[elem - 1] = 1.0
        else:
            left, right = elem
            tensor = lie_brackets(build(left), build(right))
        cache[elem] = tensor
        return tensor

    for elem in basis:
        build(elem)
    return cache


@lru_cache(maxsize=None)
def _hall_projection_matrices(width: int, depth: int) -> Dict[int, torch.Tensor]:
    _, grouped = _hall_basis_with_depths(width, depth)
    tensors = _hall_basis_tensors(width, depth)
    matrices: Dict[int, torch.Tensor] = {}

    for current_depth in range(1, depth + 1):
        elems = grouped.get(current_depth)
        if not elems:
            continue
        columns = [tensors[elem].reshape(-1) for elem in elems]
        stacked = torch.stack(columns, dim=1).to(torch.float64)  # (n^d, count)
        try:
            q, r = torch.linalg.qr(stacked, mode="reduced")
            inv_r = torch.linalg.inv(r)
            proj = inv_r @ q.transpose(0, 1)
        except Exception:
            proj = torch.linalg.pinv(stacked)
        matrices[current_depth] = proj.transpose(0, 1).contiguous()
    return matrices


@dataclass
class HallProjector:
    """Projector from tensor algebra coordinates to Hall basis coordinates.

    This class computes and caches projection matrices that convert log-signature
    tensors from tensor algebra coordinates to Hall basis coordinates. The
    projection matrices are computed using QR decomposition or pseudoinverse.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet).
    depth : int
        Truncation depth.
    device : torch.device
        Device on which to store projection matrices.
    dtype : torch.dtype
        Data type for projection matrices.

    Attributes
    ----------
    width : int
        Path dimension.
    depth : int
        Truncation depth.
    device : torch.device
        Device for projection matrices.
    dtype : torch.dtype
        Data type for projection matrices.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.hall_projection import HallProjector
    >>>
    >>> projector = HallProjector(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
    >>> # Project log-signature tensors
    >>> log_sig_tensors = [
    ...     torch.tensor([[1.0, 2.0]]),  # depth 1
    ...     torch.tensor([[[0.5, 0.3], [0.2, 0.1]]]),  # depth 2
    ... ]
    >>> result = projector.project(log_sig_tensors)
    >>> result.shape
    torch.Size([1, 3])  # logsigdim(2, 2) = 3
    """

    width: int
    depth: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        basis, grouped = _hall_basis_with_depths(self.width, self.depth)
        self._basis = list(basis)
        self._depth_offsets: Dict[int, Tuple[int, int]] = {}
        offset = 0
        for d in range(1, self.depth + 1):
            count = len(grouped.get(d, ()))
            self._depth_offsets[d] = (offset, offset + count)
            offset += count
        base_mats = _hall_projection_matrices(self.width, self.depth)
        self._matrices: Dict[int, torch.Tensor] = {
            depth: mat.to(device=self.device, dtype=self.dtype)
            for depth, mat in base_mats.items()
        }

    def project(self, log_sig_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Project log-signature tensors onto Hall basis.

        Converts log-signature tensors from tensor algebra coordinates to
        Hall basis coordinates using precomputed projection matrices.

        Parameters
        ----------
        log_sig_tensors : List[torch.Tensor]
            List of log-signature tensors in tensor algebra coordinates, where
            entry ``k`` has shape ``(batch, width, ..., width)`` with ``k+1``
            trailing width axes.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, logsigdim(width, depth))`` containing
            the log-signature in Hall basis coordinates.

        Examples
        --------
        >>> import torch
        >>> from log_signatures_pytorch.hall_projection import HallProjector
        >>>
        >>> projector = HallProjector(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
        >>> log_sig_tensors = [
        ...     torch.tensor([[1.0, 2.0]]),  # depth 1: (batch=1, width=2)
        ...     torch.tensor([[[0.5, 0.3], [0.2, 0.1]]]),  # depth 2: (batch=1, width=2, width=2)
        ... ]
        >>> result = projector.project(log_sig_tensors)
        >>> result.shape
        torch.Size([1, 3])
        """
        if not log_sig_tensors:
            return torch.zeros(0, device=self.device, dtype=self.dtype)

        batch = log_sig_tensors[0].shape[0]
        coeffs: List[torch.Tensor] = []

        for current_depth in range(1, self.depth + 1):
            start, end = self._depth_offsets[current_depth]
            count = end - start
            if count == 0:
                continue

            tensor = log_sig_tensors[current_depth - 1]
            mat = self._matrices[current_depth]
            flattened = tensor.reshape(batch, -1)
            coeffs.append(flattened @ mat)

        if not coeffs:
            return torch.zeros(batch, 0, device=self.device, dtype=self.dtype)

        return torch.cat(coeffs, dim=1)


_PROJECTOR_CACHE: Dict[Tuple[int, int, torch.device, torch.dtype], HallProjector] = {}


def get_hall_projector(
    width: int, depth: int, device: torch.device, dtype: torch.dtype
) -> HallProjector:
    """Get or create a cached Hall projector.

    Returns a cached :class:`HallProjector` instance for the given parameters.
    Projectors are cached to avoid recomputing projection matrices for the same
    width, depth, device, and dtype combination.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet).
    depth : int
        Truncation depth.
    device : torch.device
        Device on which to store projection matrices.
    dtype : torch.dtype
        Data type for projection matrices.

    Returns
    -------
    HallProjector
        A cached projector instance for the specified parameters.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.hall_projection import get_hall_projector
    >>>
    >>> # First call creates and caches the projector
    >>> projector1 = get_hall_projector(2, 2, torch.device("cpu"), torch.float32)
    >>>
    >>> # Second call returns the cached projector
    >>> projector2 = get_hall_projector(2, 2, torch.device("cpu"), torch.float32)
    >>> projector1 is projector2
    True
    """
    key = (width, depth, device, dtype)
    if key not in _PROJECTOR_CACHE:
        _PROJECTOR_CACHE[key] = HallProjector(width, depth, device, dtype)
    return _PROJECTOR_CACHE[key]
