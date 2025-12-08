import pytest
import torch

from log_signatures_pytorch.basis import (
    hall_basis,
    logsigdim,
    logsigdim_words,
    logsigkeys,
    logsigkeys_words,
)
from log_signatures_pytorch.log_signature import log_signature
from log_signatures_pytorch.signature import signature
from log_signatures_pytorch.tensor_ops import (
    batch_lie_brackets,
    batch_restricted_exp,
    lie_brackets,
)


def _basis_depth(elem) -> int:
    if isinstance(elem, int):
        return 1
    left, right = elem
    return _basis_depth(left) + _basis_depth(right)


@pytest.mark.parametrize(
    "width,depth",
    [(w, d) for w in (1, 2, 3) for d in (1, 2, 3, 4)],
)
def test_logsigdim_matches_basis_and_keys(width: int, depth: int) -> None:
    basis = hall_basis(width, depth)
    keys = logsigkeys(width, depth)
    dim = logsigdim(width, depth)

    assert dim == len(basis) == len(keys)
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("width,depth", [(2, 2), (3, 3), (3, 4)])
def test_logsigdim_words_matches_hall(width: int, depth: int) -> None:
    dim_hall = logsigdim(width, depth)
    dim_words = logsigdim_words(width, depth)
    assert dim_hall == dim_words
    keys_words = logsigkeys_words(width, depth)
    assert len(keys_words) == dim_words
    assert len(set(keys_words)) == dim_words


def test_hall_basis_is_depth_sorted_and_unique() -> None:
    basis = hall_basis(width=3, depth=4)
    depths = [_basis_depth(elem) for elem in basis]

    assert depths == sorted(depths), "basis should be ordered by increasing depth"
    assert len(basis) == len(set(basis)), "basis elements must be unique"


def test_lie_bracket_properties() -> None:
    torch.manual_seed(0)
    a = torch.randn(3, dtype=torch.float64)
    b = torch.randn(3, dtype=torch.float64)
    alpha, beta = 1.7, -0.3

    ab = lie_brackets(a, b)
    ba = lie_brackets(b, a)
    torch.testing.assert_close(ab, -ba, atol=1e-12, rtol=1e-12)

    lhs = lie_brackets(alpha * a + beta * b, b)
    rhs = alpha * lie_brackets(a, b) + beta * lie_brackets(b, b)
    torch.testing.assert_close(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_batch_lie_bracket_matches_elementwise() -> None:
    torch.manual_seed(1)
    a = torch.randn(4, 3, dtype=torch.float64)
    b = torch.randn(4, 3, dtype=torch.float64)
    batch_result = batch_lie_brackets(a, b)

    expected = torch.stack([lie_brackets(a[i], b[i]) for i in range(a.shape[0])], dim=0)
    torch.testing.assert_close(batch_result, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "width,depth,mode",
    [
        (2, 1, "hall"),
        (2, 2, "hall"),
        (3, 2, "hall"),
        (3, 3, "hall"),
        (2, 2, "words"),
        (3, 3, "words"),
    ],
)
def test_log_signature_zero_path_is_zero(width: int, depth: int, mode: str) -> None:
    path = torch.zeros(1, 5, width, dtype=torch.float64)
    log_sig = log_signature(path, depth=depth, mode=mode)
    torch.testing.assert_close(log_sig, torch.zeros_like(log_sig), atol=0, rtol=0)


def test_log_signature_single_segment_has_only_level1() -> None:
    displacement = torch.tensor([1.5, -0.2], dtype=torch.float64)
    path = torch.stack(
        [torch.zeros(2, dtype=torch.float64), displacement], dim=0
    ).unsqueeze(0)

    depth = 3
    log_sig = log_signature(path, depth=depth)
    expected_dim = logsigdim(2, depth)
    assert log_sig.shape[-1] == expected_dim

    expected = torch.zeros(1, expected_dim, dtype=log_sig.dtype)
    expected[0, 0:2] = displacement
    torch.testing.assert_close(log_sig, expected, atol=1e-10, rtol=1e-10)


def test_log_signature_single_segment_words_matches_level1() -> None:
    displacement = torch.tensor([0.3, 0.4, -0.1], dtype=torch.float64)
    path = torch.stack([torch.zeros_like(displacement), displacement], dim=0).unsqueeze(
        0
    )
    depth = 3
    log_sig = log_signature(path, depth=depth, mode="words")
    expected_dim = logsigdim_words(3, depth)
    assert log_sig.shape[-1] == expected_dim
    expected = torch.zeros(1, expected_dim, dtype=log_sig.dtype)
    expected[0, 0:3] = displacement
    torch.testing.assert_close(log_sig, expected, atol=1e-10, rtol=1e-10)


def test_log_signature_depth1_matches_signature_level1() -> None:
    torch.manual_seed(2)
    path = torch.randn(1, 6, 3, dtype=torch.float64)

    log_sig = log_signature(path, depth=1)
    sig = signature(path, depth=1)

    torch.testing.assert_close(log_sig, sig, atol=1e-12, rtol=1e-12)


def test_log_signature_grad_propagates() -> None:
    base_path = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    out = log_signature(base_path.unsqueeze(0), depth=2).sum()
    out.backward()

    assert base_path.grad is not None
    assert torch.isfinite(base_path.grad).all()
    assert not torch.isnan(base_path.grad).any()


# Helpers for signature-level verifications
def _split_signature_levels(
    sig: torch.Tensor, width: int, depth: int
) -> list[torch.Tensor]:
    """Split a flattened signature into per-level tensors shaped (width,)*k."""
    prefix_shape = sig.shape[:-1]
    levels = []
    offset = 0
    for k in range(1, depth + 1):
        size = width**k
        chunk = sig[..., offset : offset + size]
        levels.append(chunk.reshape(*prefix_shape, *([width] * k)))
        offset += size
    return levels


def _flatten_levels(levels: list[torch.Tensor]) -> torch.Tensor:
    prefix_ndim = levels[0].ndim - 1
    flat = [lvl.reshape(*lvl.shape[:prefix_ndim], -1) for lvl in levels]
    return torch.cat(flat, dim=-1)


def _chen_concatenate(
    levels_x: list[torch.Tensor], levels_y: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Compute truncated Chen product of two signatures given per-level tensors."""
    depth = len(levels_x)
    width = levels_x[0].shape[-1]
    prefix_shape = levels_x[0].shape[: levels_x[0].ndim - 1]
    one = torch.ones(prefix_shape, dtype=levels_x[0].dtype, device=levels_x[0].device)

    out: list[torch.Tensor] = []
    for k in range(1, depth + 1):
        total = torch.zeros(
            *prefix_shape,
            *([width] * k),
            dtype=levels_x[0].dtype,
            device=levels_x[0].device,
        )
        for i in range(0, k + 1):
            left = one if i == 0 else levels_x[i - 1]
            right_depth = k - i
            right = one if right_depth == 0 else levels_y[right_depth - 1]
            left_flat = left.reshape(*prefix_shape, -1)
            right_flat = right.reshape(*prefix_shape, -1)
            prod = torch.einsum("...p,...q->...pq", left_flat, right_flat)
            total = total + prod.reshape(*prefix_shape, *([width] * k))
        out.append(total)
    return out


def test_signature_translation_invariance() -> None:
    torch.manual_seed(4)
    path = torch.randn(1, 6, 3, dtype=torch.float64)
    shift = torch.tensor([2.5, -1.0, 0.3], dtype=torch.float64)

    sig = signature(path, depth=3)
    sig_shifted = signature(path + shift, depth=3)

    torch.testing.assert_close(sig, sig_shifted, atol=1e-12, rtol=1e-12)


def test_signature_single_segment_matches_restricted_exp() -> None:
    displacement = torch.tensor([0.7, -1.2], dtype=torch.float64)
    path = torch.stack([torch.zeros_like(displacement), displacement], dim=0).unsqueeze(
        0
    )
    depth = 4

    sig = signature(path, depth=depth)
    expected_levels = batch_restricted_exp(displacement.unsqueeze(0), depth=depth)
    expected = _flatten_levels(expected_levels)

    torch.testing.assert_close(sig, expected, atol=1e-12, rtol=1e-12)


def test_signature_stream_final_matches_full() -> None:
    torch.manual_seed(5)
    path = torch.randn(1, 5, 2, dtype=torch.float64)
    depth = 3

    full = signature(path, depth=depth)
    stream = signature(path, depth=depth, stream=True)

    torch.testing.assert_close(stream[:, -1], full, atol=1e-12, rtol=1e-12)


def test_signature_chen_identity_for_concatenation() -> None:
    torch.manual_seed(6)
    # Two paths that join continuously
    path1 = torch.tensor(
        [[0.0, 0.0], [0.4, -0.1], [0.9, 0.2]],
        dtype=torch.float64,
    ).unsqueeze(0)
    path2 = torch.tensor(
        [[0.9, 0.2], [1.2, 0.8], [1.6, 0.6]],
        dtype=torch.float64,
    ).unsqueeze(0)
    depth = 3

    sig1 = signature(path1, depth=depth)
    sig2 = signature(path2, depth=depth)
    sig_concat = signature(torch.cat([path1, path2[:, 1:]], dim=1), depth=depth)

    levels1 = _split_signature_levels(sig1, width=2, depth=depth)
    levels2 = _split_signature_levels(sig2, width=2, depth=depth)
    expected_levels = _chen_concatenate(levels1, levels2)
    expected = _flatten_levels(expected_levels)

    torch.testing.assert_close(sig_concat, expected, atol=1e-9, rtol=1e-9)
