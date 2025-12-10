import pytest
import torch

from log_signatures_pytorch.signature import signature, windowed_signature


def _explicit_windowed_signature(
    path: torch.Tensor, depth: int, window_size: int, hop_size: int
) -> torch.Tensor:
    """Reference implementation using explicit window batching."""
    windows = path.unfold(
        dimension=1, size=window_size, step=hop_size
    )  # (batch, num_windows, width, window_size)
    windows = windows.permute(0, 1, 3, 2)  # (batch, num_windows, window_size, width)
    batch, num_windows, _, dim = windows.shape
    flattened = windows.reshape(batch * num_windows, window_size, dim)
    sig = signature(flattened, depth=depth, stream=False)
    sigdim = sig.shape[1]
    return sig.reshape(batch, num_windows, sigdim)


@pytest.mark.parametrize("window_size,hop_size", [(16, 4), (4, 2), (5, 3)])
def test_windowed_signature_matches_explicit(window_size: int, hop_size: int):
    torch.manual_seed(0)
    batch, length, width = 2, 160, 3
    path = torch.randn(batch, length, width, dtype=torch.float64)
    depth = 3

    actual = windowed_signature(
        path, depth=depth, window_size=window_size, hop_size=hop_size
    )
    expected = _explicit_windowed_signature(
        path, depth=depth, window_size=window_size, hop_size=hop_size
    )

    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


def test_windowed_signature_single_window_equals_full_signature():
    torch.manual_seed(1)
    batch, length, width = 3, 7, 2
    path = torch.randn(batch, length, width, dtype=torch.float64)
    depth = 2

    actual = windowed_signature(path, depth=depth, window_size=length, hop_size=1)
    full = signature(path, depth=depth, stream=False)

    torch.testing.assert_close(actual.squeeze(1), full, atol=1e-9, rtol=1e-7)
