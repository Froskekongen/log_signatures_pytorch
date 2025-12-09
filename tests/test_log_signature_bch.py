import pytest
import torch

from log_signatures_pytorch.log_signature import log_signature


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_bch_matches_default_non_stream(depth: int) -> None:
    torch.manual_seed(0)
    path = torch.randn(1, 9, 3, dtype=torch.float64)
    default = log_signature(path, depth=depth, mode="hall")
    bch_sparse = log_signature(path, depth=depth, method="bch_sparse", mode="hall")
    torch.testing.assert_close(bch_sparse, default, atol=1e-9, rtol=1e-9)


def test_bch_stream_matches_default_stream() -> None:
    torch.manual_seed(1)
    depth = 3
    path = torch.randn(1, 17, 2, dtype=torch.float64)
    default = log_signature(path, depth=depth, stream=True, mode="hall")
    bch_sparse = log_signature(
        path, depth=depth, stream=True, method="bch_sparse", mode="hall"
    )
    torch.testing.assert_close(bch_sparse, default, atol=1e-9, rtol=1e-9)


def test_bch_falls_back_for_depth_above_support() -> None:
    torch.manual_seed(2)
    depth = 6
    path = torch.randn(1, 5, 2, dtype=torch.float64)
    expected = log_signature(path, depth=depth, mode="hall")
    result = log_signature(path, depth=depth, method="bch_sparse", mode="hall")
    torch.testing.assert_close(result, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize(
    ("stream", "seed", "shape"),
    [
        (False, 3, (1, 13, 5)),
        (True, 4, (1, 9, 3)),
    ],
)
def test_bch_sparse_grad_propagates_param(
    stream: bool, seed: int, shape: tuple[int, ...]
) -> None:
    torch.manual_seed(seed)
    path = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    out = log_signature(path, depth=3, stream=stream, method="bch_sparse", mode="hall")
    loss = out.sum()
    loss.backward()
    assert path.grad is not None
    assert torch.isfinite(path.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("stream", [False, True])
def test_bch_sparse_grad_propagates_cuda(stream: bool) -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    path = torch.randn(1, 6, 2, device=device, dtype=torch.float32, requires_grad=True)
    out = log_signature(path, depth=3, stream=stream, method="bch_sparse", mode="hall")
    loss = out.sum()
    loss.backward()
    assert path.grad is not None
    assert torch.isfinite(path.grad).all()
