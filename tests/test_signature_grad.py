import pytest
import torch

from log_signatures_pytorch.signature import signature


@pytest.mark.parametrize(
    ("stream", "seed", "shape", "depth"),
    [
        (False, 0, (2, 5, 3), 3),
        (True, 1, (1, 4, 2), 2),
    ],
)
def test_signature_grad_propagates(
    stream: bool, seed: int, shape: tuple[int, ...], depth: int
) -> None:
    torch.manual_seed(seed)
    path = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    out = signature(path, depth=depth, stream=stream)
    out.sum().backward()

    assert path.grad is not None
    assert torch.isfinite(path.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("stream", [False, True])
def test_signature_grad_propagates_cuda(stream: bool) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    path = torch.randn(1, 3, 2, device=device, dtype=torch.float32, requires_grad=True)

    out = signature(path, depth=2, stream=stream)
    out.sum().backward()

    assert path.grad is not None
    assert torch.isfinite(path.grad).all()
