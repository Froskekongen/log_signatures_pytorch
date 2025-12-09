"""Hall-basis BCH utilities for incremental log-signature computation.

This module provides utilities for computing log-signatures using the
Baker-Campbell-Hausdorff formula directly in Hall basis coordinates,
avoiding the need to compute the full signature first.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from .hall_projection import (
    hall_basis,
    logsigdim,
    _hall_basis_tensors,
    _hall_element_depth,
    get_hall_projector,
)
from .tensor_ops import lie_brackets


@lru_cache(maxsize=None)
def _structure_constants(width: int, depth: int) -> torch.Tensor:
    """Pre-compute [e_i, e_j] expansion in the Hall basis.

    Computes the structure constants of the free Lie algebra in the Hall basis,
    which encode how Lie brackets of basis elements expand in the basis.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet).
    depth : int
        Truncation depth.

    Returns
    -------
    torch.Tensor
        Tensor ``C`` of shape ``(dim, dim, dim)`` such that
        ``bracket(v, w) = einsum('i,j,ijk->k', v, w, C)`` for Hall-coordinate
        vectors ``v`` and ``w``, where ``dim = logsigdim(width, depth)``.

    Notes
    -----
    This function is cached to avoid recomputing structure constants for the
    same width and depth combination.
    """
    basis = hall_basis(width, depth)
    dim = len(basis)
    constants = torch.zeros(dim, dim, dim, dtype=torch.float64)
    projector = get_hall_projector(
        width=width,
        depth=depth,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    basis_tensors = _hall_basis_tensors(width, depth)

    zeros_template = [
        torch.zeros((1, *([width] * current_depth)), dtype=torch.float64)
        for current_depth in range(1, depth + 1)
    ]

    for i, elem_i in enumerate(basis):
        depth_i = _hall_element_depth(elem_i)
        tensor_i = basis_tensors[elem_i]
        for j, elem_j in enumerate(basis):
            total_depth = depth_i + _hall_element_depth(elem_j)
            if total_depth > depth:
                continue
            bracket_tensor = lie_brackets(tensor_i, basis_tensors[elem_j])
            log_sig_tensors = [t.clone() for t in zeros_template]
            log_sig_tensors[total_depth - 1] = bracket_tensor.unsqueeze(0)
            coeffs = projector.project(log_sig_tensors).squeeze(0)
            constants[i, j] = coeffs
    return constants


@lru_cache(maxsize=None)
def _sparse_constants(width: int, depth: int):
    dense = _structure_constants(width, depth)
    idx = torch.nonzero(dense, as_tuple=False)  # (nnz, 3) over (i, j, k)
    values = dense[idx[:, 0], idx[:, 1], idx[:, 2]]
    return idx, values


class HallBCH:
    """Truncated BCH on Hall-basis coordinates (supports depth <= 4).

    This class implements the Baker-Campbell-Hausdorff formula directly in
    Hall basis coordinates, allowing incremental log-signature computation
    without materializing the full tensor-algebra signature.

    Parameters
    ----------
    width : int
        Path dimension (size of the alphabet).
    depth : int
        Truncation depth. Must be <= 4 for exact computation.
    device : torch.device
        Device on which to store structure constants and perform computations.
    dtype : torch.dtype
        Data type for computations.

    Attributes
    ----------
    width : int
        Path dimension.
    depth : int
        Truncation depth.
    dim : int
        Dimension of the log-signature (logsigdim(width, depth)).
    device : torch.device
        Device for computations.
    dtype : torch.dtype
        Data type for computations.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.hall_bch import HallBCH
    >>>
    >>> bch = HallBCH(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
    >>> x = torch.tensor([[1.0, 2.0, 0.0]])  # Hall coordinates (batch=1, dim=3)
    >>> y = torch.tensor([[3.0, 4.0, 0.0]])  # Hall coordinates (batch=1, dim=3)
    >>> result = bch.bch(x, y)
    >>> result.shape
    torch.Size([1, 3])
    """

    def __init__(
        self,
        width: int,
        depth: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.width = width
        self.depth = depth
        self.dim = logsigdim(width, depth)
        sparse_idx, sparse_vals = _sparse_constants(width, depth)
        self._sparse_idx = sparse_idx.to(device=device)
        self._sparse_vals = sparse_vals.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def increment_to_hall(self, delta: torch.Tensor) -> torch.Tensor:
        """Embed path increment (batch, width) into Hall coordinates.

        Converts a path increment (which lives in degree-1 of the free Lie algebra)
        into Hall basis coordinates by placing it in the first ``width`` components
        and zeroing the rest.

        Parameters
        ----------
        delta : torch.Tensor
            Tensor of shape ``(batch, width)`` representing path increments.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, self.dim)`` with the increment embedded
            in Hall coordinates.

        Examples
        --------
        >>> import torch
        >>> from log_signatures_pytorch.hall_bch import HallBCH
        >>>
        >>> bch = HallBCH(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
        >>> delta = torch.tensor([[1.0, 2.0]])  # (batch=1, width=2)
        >>> result = bch.increment_to_hall(delta)
        >>> result.shape
        torch.Size([1, 3])
        >>> result[:, :2]  # First two components match delta
        tensor([[1., 2.]])
        """
        batch = delta.shape[0]
        out = torch.zeros(batch, self.dim, device=self.device, dtype=self.dtype)
        out[:, : self.width] = delta
        return out

    def bracket_sparse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sparse scatter-based bracket using nonzero structure constants.

        Computes the Lie bracket [x, y] in Hall coordinates using sparse
        structure constants for efficiency.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, self.dim)`` in Hall coordinates.
        y : torch.Tensor
            Tensor of shape ``(batch, self.dim)`` in Hall coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, self.dim)`` representing [x, y] in
            Hall coordinates.

        Notes
        -----
        This implementation uses sparse structure constants to avoid computing
        all possible bracket combinations, improving efficiency.

        Examples
        --------
        >>> import torch
        >>> from log_signatures_pytorch.hall_bch import HallBCH
        >>>
        >>> bch = HallBCH(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
        >>> x = torch.tensor([[1.0, 2.0, 0.0]])
        >>> y = torch.tensor([[3.0, 4.0, 0.0]])
        >>> bracket = bch.bracket_sparse(x, y)
        >>> bracket.shape
        torch.Size([1, 3])
        """
        if self._sparse_idx.numel() == 0:
            return torch.zeros_like(x)
        i_idx = self._sparse_idx[:, 0]
        j_idx = self._sparse_idx[:, 1]
        k_idx = self._sparse_idx[:, 2]
        coeff = self._sparse_vals
        # Gather x_i and y_j for all nonzeros
        xi = x[:, i_idx]  # (batch, nnz)
        yj = y[:, j_idx]  # (batch, nnz)
        contrib = xi * yj * coeff  # broadcast over batch
        out = torch.zeros_like(x)
        out.scatter_add_(1, k_idx.unsqueeze(0).expand_as(contrib), contrib)
        return out

    def bch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """BCH(x, y) truncated to depth <= 4 (caps at 4 if configured higher).

        Computes the Baker-Campbell-Hausdorff formula BCH(x, y) = log(exp(x) exp(y))
        in Hall coordinates, truncated to the specified depth.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch, self.dim)`` in Hall coordinates.
        y : torch.Tensor
            Tensor of shape ``(batch, self.dim)`` in Hall coordinates.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, self.dim)`` representing BCH(x, y) in
            Hall coordinates.

        Notes
        -----
        This method supports depths up to 4. If a higher depth is configured,
        it caps the computation at depth 4. The caller should use the default
        signature→log path for higher depths.

        Examples
        --------
        >>> import torch
        >>> from log_signatures_pytorch.hall_bch import HallBCH
        >>>
        >>> bch = HallBCH(width=2, depth=2, device=torch.device("cpu"), dtype=torch.float32)
        >>> x = torch.tensor([[1.0, 2.0, 0.0]])
        >>> y = torch.tensor([[3.0, 4.0, 0.0]])
        >>> result = bch.bch(x, y)
        >>> result.shape
        torch.Size([1, 3])
        """
        depth = self.depth
        bracket = self.bracket_sparse
        if depth <= 4:
            return self._bch_closed_form(x, y, bracket, depth)
        # Unsupported depth; caller should have fallen back to default path.
        return self._bch_closed_form(x, y, bracket, 4)

    def _bch_closed_form(self, x, y, bracket, depth):
        z = x + y
        if depth == 1:
            return z
        xy = bracket(x, y)
        z = z + 0.5 * xy
        if depth == 2:
            return z
        x_xy = bracket(x, xy)
        y_yx = bracket(y, bracket(y, x))
        z = z + (1.0 / 12.0) * (x_xy + y_yx)
        if depth == 3:
            return z
        y_xxy = bracket(y, x_xy)
        z = z - (1.0 / 24.0) * y_xxy
        return z


def supports_depth(depth: int) -> bool:
    """Return True if the BCH truncation is implemented for this depth.

    Checks whether the sparse Hall-BCH method supports the given depth.
    Currently, only depths up to 4 are supported.

    Parameters
    ----------
    depth : int
        Truncation depth to check.

    Returns
    -------
    bool
        True if depth <= 4, False otherwise.

    Examples
    --------
    >>> from log_signatures_pytorch.hall_bch import supports_depth
    >>>
    >>> supports_depth(2)
    True
    >>> supports_depth(4)
    True
    >>> supports_depth(5)
    False
    """
    return depth <= 4
