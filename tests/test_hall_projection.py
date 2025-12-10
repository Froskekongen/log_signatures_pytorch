import torch

from log_signatures_pytorch.hall_projection import (
    hall_basis,
    logsigdim,
    _hall_basis_tensors,
    _hall_element_depth,
    get_hall_projector,
)


def _empty_log_sig(width: int, depth: int, dtype: torch.dtype) -> list[torch.Tensor]:
    tensors = []
    for current_depth in range(1, depth + 1):
        shape = [1] + [width] * current_depth
        tensors.append(torch.zeros(shape, dtype=dtype))
    return tensors


def test_projector_recovers_single_basis_element() -> None:
    width, depth = 2, 3
    dtype = torch.float64
    projector = get_hall_projector(width, depth, torch.device("cpu"), dtype)
    basis = hall_basis(width, depth)
    tensors = _hall_basis_tensors(width, depth)

    for idx, elem in enumerate(basis):
        log_sig_tensors = _empty_log_sig(width, depth, dtype)
        elem_tensor = tensors[elem].to(dtype=dtype).unsqueeze(0)
        elem_depth = _hall_element_depth(elem)
        log_sig_tensors[elem_depth - 1] = elem_tensor
        coeffs = projector.project(log_sig_tensors)
        expected = torch.zeros(1, logsigdim(width, depth), dtype=dtype)
        expected[0, idx] = 1.0
        torch.testing.assert_close(coeffs, expected, atol=1e-8, rtol=1e-8)


def test_projector_preserves_linear_combination() -> None:
    width, depth = 2, 3
    dtype = torch.float64
    projector = get_hall_projector(width, depth, torch.device("cpu"), dtype)
    basis = hall_basis(width, depth)
    tensors = _hall_basis_tensors(width, depth)
    coefficients = torch.linspace(-0.3, 0.3, steps=len(basis), dtype=dtype)

    log_sig_tensors = _empty_log_sig(width, depth, dtype)
    for coeff, elem in zip(coefficients, basis):
        elem_depth = _hall_element_depth(elem)
        elem_tensor = tensors[elem].to(dtype=dtype).unsqueeze(0)
        log_sig_tensors[elem_depth - 1] = (
            log_sig_tensors[elem_depth - 1] + coeff * elem_tensor
        )

    projected = projector.project(log_sig_tensors)
    expected = coefficients.unsqueeze(0)
    torch.testing.assert_close(projected, expected, atol=1e-8, rtol=1e-8)
