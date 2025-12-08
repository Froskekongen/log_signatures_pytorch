import pytest
import torch

esig = pytest.importorskip("esig")

from log_signatures_pytorch.log_signature import log_signature


def _random_path(
    length: int, width: int, dtype: torch.dtype, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    increments = torch.randn(length, width, dtype=dtype)
    return torch.cumsum(increments, dim=0)


@pytest.mark.external
@pytest.mark.parametrize("length", [64, 128, 256])
def test_parity_long_width2_depth4(length: int) -> None:
    depth = 4
    width = 2
    batch = 2
    dtype = torch.float64
    paths = torch.stack(
        [_random_path(length, width, dtype, seed=10 + i) for i in range(batch)],
        dim=0,
    )
    ours = log_signature(paths, depth=depth).detach()
    expected = torch.stack(
        [
            torch.tensor(esig.stream2logsig(path.numpy(), depth), dtype=dtype)
            for path in paths
        ],
        dim=0,
    )
    torch.testing.assert_close(ours, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.external
@pytest.mark.parametrize("length", [64, 128, 256])
def test_parity_long_width3_depth3(length: int) -> None:
    depth = 3
    width = 3
    batch = 1
    dtype = torch.float64
    paths = torch.stack(
        [_random_path(length, width, dtype, seed=20 + i) for i in range(batch)],
        dim=0,
    )
    ours = log_signature(paths, depth=depth).detach()
    expected = torch.stack(
        [
            torch.tensor(esig.stream2logsig(path.numpy(), depth), dtype=dtype)
            for path in paths
        ],
        dim=0,
    )
    torch.testing.assert_close(ours, expected, atol=1e-6, rtol=1e-6)
