"""Tensor algebra operations for signature and log-signature computation.

This module provides low-level tensor operations used throughout the signature
and log-signature computation, including tensor products, Lie brackets, and
truncated exponentials.
"""

import torch
from torch import Tensor


def _broadcast_tensor_product(x: Tensor, y: Tensor, shared_dims: int) -> Tensor:
    """Broadcast-aware tensor product preserving the first shared_dims axes."""
    if x.ndim < shared_dims or y.ndim < shared_dims:
        msg = (
            f"Expected at least {shared_dims} shared dimensions, "
            f"got x.ndim={x.ndim}, y.ndim={y.ndim}"
        )
        raise ValueError(msg)
    xdim = x.ndim
    ydim = y.ndim
    x_view = x.reshape(*x.shape, *([1] * (ydim - shared_dims)))
    y_view = y.reshape(
        *y.shape[:shared_dims],
        *([1] * (xdim - shared_dims)),
        *y.shape[shared_dims:],
    )
    return x_view * y_view


def tensor_product(x: Tensor, y: Tensor) -> Tensor:
    """Compute the tensor product x ⊗ y with no shared leading axes.

    For tensors x in V^{⊗p} and y in V^{⊗q}, returns x ⊗ y in V^{⊗(p+q)} by
    forming the outer product over their trailing axes. This is the
    multiplicative structure used throughout the tensor-algebra signature
    recurrences.

    Parameters
    ----------
    x : Tensor
        First tensor operand.
    y : Tensor
        Second tensor operand.

    Returns
    -------
    Tensor
        Tensor product x ⊗ y with shape ``(*x.shape, *y.shape)``.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import tensor_product
    >>>
    >>> x = torch.tensor([1.0, 2.0])
    >>> y = torch.tensor([3.0, 4.0])
    >>> result = tensor_product(x, y)
    >>> result.shape
    torch.Size([2, 2])
    >>> result
    tensor([[3., 4.],
            [6., 8.]])
    """
    return _broadcast_tensor_product(x, y, shared_dims=0)


def batch_tensor_product(x: Tensor, y: Tensor) -> Tensor:
    """Tensor product preserving the leading batch axis.

    Computes the tensor product while preserving the first (batch) dimension,
    allowing batched tensor algebra operations.

    Parameters
    ----------
    x : Tensor
        Tensor shaped ``(batch, ...)``.
    y : Tensor
        Tensor shaped ``(batch, ...)``.

    Returns
    -------
    Tensor
        Tensor shaped ``(batch, *x.shape[1:], *y.shape[1:])``.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_tensor_product
    >>>
    >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (batch=2, width=2)
    >>> y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # (batch=2, width=2)
    >>> result = batch_tensor_product(x, y)
    >>> result.shape
    torch.Size([2, 2, 2])
    """
    return _broadcast_tensor_product(x, y, shared_dims=1)


def batch_sequence_tensor_product(x: Tensor, y: Tensor) -> Tensor:
    """Tensor product preserving leading (batch, sequence) axes.

    Computes the tensor product while preserving the first two dimensions
    (batch and sequence), allowing per-step tensor products in sequence
    processing.

    Parameters
    ----------
    x : Tensor
        Tensor shaped ``(batch, sequence, ...)``.
    y : Tensor
        Tensor shaped ``(batch, sequence, ...)``.

    Returns
    -------
    Tensor
        Tensor shaped ``(batch, sequence, *x.shape[2:], *y.shape[2:])``.

    Notes
    -----
    This is used by the GPU signature scan where per-step products are formed
    without collapsing the time dimension.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_sequence_tensor_product
    >>>
    >>> x = torch.randn(2, 3, 2)  # (batch=2, sequence=3, width=2)
    >>> y = torch.randn(2, 3, 2)  # (batch=2, sequence=3, width=2)
    >>> result = batch_sequence_tensor_product(x, y)
    >>> result.shape
    torch.Size([2, 3, 2, 2])
    """
    return _broadcast_tensor_product(x, y, shared_dims=2)


def _add_tensor_product(x: Tensor, y: Tensor, z: Tensor, shared_dims: int) -> Tensor:
    """Compute x + y ⊗ z while respecting shared leading dimensions."""
    return x + _broadcast_tensor_product(y, z, shared_dims=shared_dims)


def add_tensor_product(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Affine tensor product x + y ⊗ z in the tensor algebra.

    Computes the sum of tensor x and the tensor product of y and z.

    Parameters
    ----------
    x : Tensor
        First tensor operand.
    y : Tensor
        Second tensor operand (first factor of product).
    z : Tensor
        Third tensor operand (second factor of product).

    Returns
    -------
    Tensor
        Result of x + y ⊗ z.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import add_tensor_product
    >>>
    >>> x = torch.tensor([1.0, 2.0])
    >>> y = torch.tensor([3.0, 4.0])
    >>> z = torch.tensor([5.0, 6.0])
    >>> result = add_tensor_product(x, y, z)
    >>> result.shape
    torch.Size([2, 2])
    """
    return _add_tensor_product(x, y, z, shared_dims=0)


def batch_add_tensor_product(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Batched version of x + y ⊗ z preserving the leading batch axis.

    Computes the affine tensor product while preserving the first (batch)
    dimension.

    Parameters
    ----------
    x : Tensor
        Tensor shaped ``(batch, ...)``.
    y : Tensor
        Tensor shaped ``(batch, ...)``.
    z : Tensor
        Tensor shaped ``(batch, ...)``.

    Returns
    -------
    Tensor
        Tensor shaped ``(batch, *x.shape[1:], *y.shape[1:], *z.shape[1:])``
        (after broadcasting).

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_add_tensor_product
    >>>
    >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (batch=2, width=2)
    >>> y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # (batch=2, width=2)
    >>> z = torch.tensor([[9.0, 10.0], [11.0, 12.0]])  # (batch=2, width=2)
    >>> result = batch_add_tensor_product(x, y, z)
    >>> result.shape
    torch.Size([2, 2, 2])
    """
    return _add_tensor_product(x, y, z, shared_dims=1)


def batch_restricted_exp(input: Tensor, depth: int) -> list[Tensor]:
    """Batched truncated tensor exponential with a shared batch axis.

    Computes the truncated tensor exponential exp(input) - 1 for batched inputs,
    returning homogeneous components at each depth level. Each batch element
    receives the homogeneous components, enabling efficient signature scans
    over a batch of paths.

    Parameters
    ----------
    input : Tensor
        Tensor of shape ``(batch, width)`` representing degree-1 elements.
    depth : int
        Truncation depth (>=1).

    Returns
    -------
    list[Tensor]
        List of length ``depth`` where entry ``k`` has shape
        ``(batch, width**(k+1))`` when flattened or ``(batch, width, ..., width)``
        with ``k+1`` trailing width dimensions.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_restricted_exp
    >>>
    >>> input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (batch=2, width=2)
    >>> result = batch_restricted_exp(input_tensor, depth=2)
    >>> len(result)
    2
    >>> result[0].shape  # depth 1
    torch.Size([2, 2])
    >>> result[1].shape  # depth 2
    torch.Size([2, 4])
    """
    ret = [input]
    for i in range(2, depth + 1):
        ret.append(_broadcast_tensor_product(ret[-1], input / i, shared_dims=1))
    return ret


def batch_mult_fused_restricted_exp(z: Tensor, A: list[Tensor]) -> list[Tensor]:
    """Batched fused update of truncated tensor exponentials.

    Updates a list of truncated exponential terms by multiplying with a new
    degree-1 element. This is used in the signature scan to incrementally
    update signatures as we process path increments.

    Parameters
    ----------
    z : Tensor
        Tensor of shape ``(batch, n_features)`` living in degree 1,
        representing a path increment.
    A : list[Tensor]
        List of current exponential terms; ``A[k]`` has shape
        ``(batch, n_features ** (k + 1))``.

    Returns
    -------
    list[Tensor]
        Updated list of tensors with the same shapes as ``A``, representing
        the exponential terms after multiplication by exp(z).

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_mult_fused_restricted_exp
    >>>
    >>> z = torch.tensor([[1.0, 2.0]])  # (batch=1, width=2)
    >>> A = [
    ...     torch.tensor([[1.0, 2.0]]),  # depth 1: (batch=1, width=2)
    ...     torch.tensor([[0.5, 0.3, 0.2, 0.1]]),  # depth 2: (batch=1, width=4)
    ... ]
    >>> result = batch_mult_fused_restricted_exp(z, A)
    >>> len(result)
    2
    >>> result[0].shape
    torch.Size([1, 2])
    >>> result[1].shape
    torch.Size([1, 4])
    """
    depth = len(A)
    dtype = z.dtype
    device = z.device
    divisors = torch.arange(1, depth + 1, dtype=dtype, device=device).flip(0)
    divisor_view = divisors.reciprocal().view(-1, *([1] * z.ndim))
    z_divided_full = z.unsqueeze(0) * divisor_view
    unit = torch.ones(*z.shape[:1], dtype=dtype, device=device)
    ret: list[Tensor] = []
    for depth_index in range(depth):
        current = unit
        z_divided = z_divided_full[depth - depth_index - 1 :]
        for i in range(depth_index + 1):
            current = _add_tensor_product(
                x=A[i], y=current, z=z_divided[i], shared_dims=1
            )
        ret.append(current)
    return ret


def lie_brackets(x: Tensor, y: Tensor) -> Tensor:
    """Lie bracket [x, y] = x ⊗ y - y ⊗ x for degree-1 tensors.

    Computes the Lie bracket (commutator) of two degree-1 tensors, which is
    the antisymmetric part of their tensor product.

    Parameters
    ----------
    x : Tensor
        First tensor operand (degree-1).
    y : Tensor
        Second tensor operand (degree-1).

    Returns
    -------
    Tensor
        Lie bracket [x, y] = x ⊗ y - y ⊗ x.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import lie_brackets
    >>>
    >>> x = torch.tensor([1.0, 2.0])
    >>> y = torch.tensor([3.0, 4.0])
    >>> result = lie_brackets(x, y)
    >>> result.shape
    torch.Size([2, 2])
    >>> # Result is antisymmetric: [x, y] = -[y, x]
    >>> lie_brackets(y, x) + result
    tensor([[0., 0.],
            [0., 0.]])
    """
    return tensor_product(x, y) - tensor_product(y, x)


def batch_lie_brackets(x: Tensor, y: Tensor) -> Tensor:
    """Batched Lie bracket preserving the leading batch axis.

    Computes the Lie bracket for batched tensors while preserving the first
    (batch) dimension.

    Parameters
    ----------
    x : Tensor
        Tensor shaped ``(batch, ...)``.
    y : Tensor
        Tensor shaped ``(batch, ...)``.

    Returns
    -------
    Tensor
        Batched Lie bracket [x, y] with shape preserving the batch dimension.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_lie_brackets
    >>>
    >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (batch=2, width=2)
    >>> y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # (batch=2, width=2)
    >>> result = batch_lie_brackets(x, y)
    >>> result.shape
    torch.Size([2, 2, 2])
    """
    return batch_tensor_product(x, y) - batch_tensor_product(y, x)


def batch_bch_formula(a: list[Tensor], b: list[Tensor], depth: int) -> list[Tensor]:
    """Truncated Baker-Campbell-Hausdorff merge for batched inputs.

    Computes a truncated version of the Baker-Campbell-Hausdorff formula
    BCH(a, b) for batched inputs in tensor algebra coordinates.

    Parameters
    ----------
    a : list[Tensor]
        List of tensors where ``a[k]`` has shape ``(batch, width**(k+1))``,
        representing the first log-signature.
    b : list[Tensor]
        List of tensors with the same structure as ``a``, representing the
        second log-signature.
    depth : int
        Truncation depth for the BCH series.

    Returns
    -------
    list[Tensor]
        List of tensors matching the shapes of ``a``/``b`` up to ``depth``,
        representing the merged log-signature.

    Notes
    -----
    This implementation includes only the series terms it explicitly writes:
    a + b for all depths, and + 1/2 [a, b] when depth >= 2. Higher-order BCH
    terms are not included. For richer Hall-basis truncations up to depth 4,
    use :meth:`HallBCH.bch`.

    Examples
    --------
    >>> import torch
    >>> from log_signatures_pytorch.tensor_ops import batch_bch_formula
    >>>
    >>> a = [
    ...     torch.tensor([[1.0, 2.0]]),  # depth 1
    ...     torch.tensor([[0.5, 0.3, 0.2, 0.1]]),  # depth 2
    ... ]
    >>> b = [
    ...     torch.tensor([[3.0, 4.0]]),  # depth 1
    ...     torch.tensor([[0.6, 0.4, 0.3, 0.2]]),  # depth 2
    ... ]
    >>> result = batch_bch_formula(a, b, depth=2)
    >>> len(result)
    2
    >>> result[0].shape
    torch.Size([1, 2])
    """
    if depth == 0:
        return []

    if not a and not b:
        msg = "BCH merge requires at least one operand."
        raise ValueError(msg)

    reference = a[0] if a else b[0]
    dtype = reference.dtype
    device = reference.device
    batch_size = reference.shape[0]
    width = reference.shape[1]

    result: list[Tensor] = []
    for idx in range(depth):
        if idx < len(a) and idx < len(b):
            result.append(a[idx] + b[idx])
        elif idx < len(a):
            result.append(a[idx].clone())
        elif idx < len(b):
            result.append(b[idx].clone())
        else:
            shape = (batch_size,) + (width,) * (idx + 1)
            result.append(torch.zeros(shape, dtype=dtype, device=device))

    if depth >= 2 and a and b:
        bracket = batch_lie_brackets(a[0], b[0]) / 2.0
        if len(result) >= 2:
            result[1] = result[1] + bracket
        else:
            result.append(bracket)
    return result
