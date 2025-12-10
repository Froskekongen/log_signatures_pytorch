import pytest
import torch

roughpy = pytest.importorskip("roughpy")

from log_signatures_pytorch.log_signature import log_signature
from log_signatures_pytorch.bch_coefficients import bch_coeffs
from log_signatures_pytorch.hall_projection import get_hall_projector


def free_tensor_logsig(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Compute log-signature via roughpy and project to Hall basis."""
    batch, length, width = path.shape
    assert batch == 1, "helper assumes batch=1"
    ctx = roughpy.get_context(width=width, depth=depth, coeffs=roughpy.DPReal)
    # Build signature as product of exp of increments
    sig = roughpy.FreeTensor(1.0, ctx=ctx)
    for inc in torch.diff(path[0], dim=0):
        comp = {
            roughpy.TensorKey((i + 1,), ctx=ctx): float(inc[i].item())
            for i in range(width)
        }
        elem = roughpy.FreeTensor(comp, ctx=ctx).exp()
        sig = sig * elem
    log_elem = sig.log()
    # Build tensor components per degree
    tensors = []
    device = path.device
    dtype = path.dtype
    for d in range(1, depth + 1):
        tensors.append(torch.zeros([width] * d, dtype=dtype, device=device))
    for item in log_elem:
        word = tuple(item.key().to_letters())
        if len(word) > depth:
            continue
        coeff = item.value().to_float()
        idx = tuple(i - 1 for i in word)
        tensors[len(word) - 1][idx] = coeff
    projector = get_hall_projector(width, depth, device=device, dtype=dtype)
    coeffs = projector.project([t.unsqueeze(0) for t in tensors]).squeeze(0)
    return coeffs.unsqueeze(0)


def bch_words(depth: int, width: int = 2) -> dict[tuple[int, ...], float]:
    ctx = roughpy.get_context(width=width, depth=depth, coeffs=roughpy.DPReal)
    elems = [
        roughpy.FreeTensor({roughpy.TensorKey((i + 1,), ctx=ctx): 1.0}, ctx=ctx).exp()
        for i in range(width)
    ]
    prod = elems[0]
    for e in elems[1:]:
        prod = prod * e
    log_elem = prod.log()
    coeffs: dict[tuple[int, ...], float] = {}
    for item in log_elem:
        word = tuple(item.key().to_letters())
        if len(word) > depth:
            continue
        coeffs[word] = item.value().to_float()
    return coeffs


@pytest.mark.external
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_bch_sparse_matches_roughpy(depth: int) -> None:
    torch.manual_seed(0)
    width = 3
    path = torch.randn(1, 3, width, dtype=torch.float64)
    ours = log_signature(path, depth=depth, method="bch_sparse", mode="hall")
    ref = free_tensor_logsig(path, depth=depth)
    torch.testing.assert_close(ours, ref, atol=1e-8, rtol=1e-8)


@pytest.mark.external
@pytest.mark.parametrize(
    "width,depth", [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]
)
def test_sympy_dynkin_matches_roughpy(width: int, depth: int) -> None:
    coeffs_sym = bch_coeffs(width=width, depth=depth)
    coeffs_rough = bch_words(depth, width=width)
    for word, coeff in coeffs_sym.items():
        assert word in coeffs_rough
        assert float(coeff) == pytest.approx(coeffs_rough[word])
