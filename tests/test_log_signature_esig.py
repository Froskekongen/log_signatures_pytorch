import random

import pytest
import torch

esig = pytest.importorskip("esig")

from log_signatures_pytorch.log_signature import log_signature
from log_signatures_pytorch.hall_projection import logsigkeys, logsigdim


def _esig_keys(width: int, depth: int) -> list[str]:
    raw = esig.logsigkeys(width, depth)
    return [k for k in raw.split(" ") if k]


def _random_path(length: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    increments = torch.randn(length, width, dtype=dtype)
    return torch.cumsum(increments, dim=0)


@pytest.mark.external
@pytest.mark.parametrize("depth", [2, 3, 4])
def test_log_signature_matches_esig_single_path(depth: int) -> None:
    torch.manual_seed(0)
    width = 2
    length = random.randint(10, 30)
    path = _random_path(length, width, torch.float64)
    ours = log_signature(path.unsqueeze(0), depth=depth, mode="hall").detach()
    esig_result = torch.tensor(
        esig.stream2logsig(path.numpy(), depth), dtype=ours.dtype
    ).unsqueeze(0)
    torch.testing.assert_close(ours, esig_result, atol=1e-6, rtol=1e-6)


@pytest.mark.external
def test_logsigkeys_and_dim_match_esig() -> None:
    for width in [2, 3]:
        for depth in [2, 3, 4]:
            ours_keys = logsigkeys(width, depth)
            theirs_keys = _esig_keys(width, depth)
            assert ours_keys == theirs_keys
            assert logsigdim(width, depth) == esig.logsigdim(width, depth)


@pytest.mark.external
def test_depth4_batched_matches_esig_width3() -> None:
    torch.manual_seed(2)
    width = 3
    depth = 4
    batch = 2
    length = 18
    paths = torch.stack(
        [_random_path(length, width, torch.float64) for _ in range(batch)], dim=0
    )
    ours = log_signature(paths, depth=depth, mode="hall").detach()
    expected = torch.stack(
        [
            torch.tensor(
                esig.stream2logsig(path.numpy(), depth),
                dtype=ours.dtype,
            )
            for path in paths
        ],
        dim=0,
    )
    torch.testing.assert_close(ours, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.external
@pytest.mark.parametrize("depth", [2, 3])
def test_batched_log_signature_matches_esig(depth: int) -> None:
    torch.manual_seed(1)
    width = 3
    batch = 3
    length = 12
    paths = torch.stack(
        [_random_path(length, width, torch.float64) for _ in range(batch)],
        dim=0,
    )
    ours = log_signature(paths, depth=depth, mode="hall").detach()
    expected = torch.stack(
        [
            torch.tensor(
                esig.stream2logsig(path.numpy(), depth),
                dtype=ours.dtype,
            )
            for path in paths
        ],
        dim=0,
    )
    torch.testing.assert_close(ours, expected, atol=1e-6, rtol=1e-6)
