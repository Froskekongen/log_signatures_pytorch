import pytest
import torch

from log_signatures_pytorch.log_signature import log_signature, windowed_log_signature


def _explicit_windowed_log_signature(
    path: torch.Tensor, depth: int, window_size: int, hop_size: int, mode: str = "words"
) -> torch.Tensor:
    """Baseline using torch.unfold + batched log_signature."""
    windows = path.unfold(dimension=1, size=window_size, step=hop_size)  # (batch, num_windows, width, window)
    windows = windows.permute(0, 1, 3, 2)  # (batch, num_windows, window, width)
    batch, num_windows, _, width = windows.shape
    flattened = windows.reshape(batch * num_windows, window_size, width)
    logsig = log_signature(flattened, depth=depth, mode=mode)
    return logsig.reshape(batch, num_windows, -1)


@pytest.mark.parametrize("mode", ["words", "hall"])
@pytest.mark.parametrize("window_size,hop_size", [(32, 8), (3, 1), (4, 2)])
def test_windowed_log_signature_matches_explicit(mode: str, window_size: int, hop_size: int):
    torch.manual_seed(0)
    batch, length, width = 2, 160, 3
    path = torch.randn(batch, length, width, dtype=torch.float64)
    depth = 3

    actual = windowed_log_signature(
        path, depth=depth, window_size=window_size, hop_size=hop_size, mode=mode
    )
    expected = _explicit_windowed_log_signature(
        path, depth=depth, window_size=window_size, hop_size=hop_size, mode=mode
    )

    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


def test_windowed_log_signature_single_window_equals_full():
    torch.manual_seed(1)
    batch, length, width = 3, 7, 2
    path = torch.randn(batch, length, width, dtype=torch.float64)
    depth = 2

    actual = windowed_log_signature(path, depth=depth, window_size=length, hop_size=1)
    full = log_signature(path, depth=depth)

    torch.testing.assert_close(actual.squeeze(1), full, atol=1e-9, rtol=1e-7)
