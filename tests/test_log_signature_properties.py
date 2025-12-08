import torch

from log_signatures_pytorch.log_signature import log_signature


def test_single_segment_logsig_is_increment():
    path = torch.tensor([[0.0, 0.0], [1.5, -0.5]], dtype=torch.double).unsqueeze(0)
    logsig = log_signature(path, depth=3)
    expected = torch.tensor([[1.5, -0.5, 0.0, 0.0, 0.0]], dtype=torch.double)
    assert torch.allclose(logsig, expected)


def test_two_segment_area_matches_bracket():
    path = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    ).unsqueeze(0)
    logsig = log_signature(path, depth=2)
    expected = torch.tensor([[1.0, 1.0, 0.5]], dtype=torch.double)
    assert torch.allclose(logsig, expected)


def test_stream_prefix_matches_full_logsig():
    path = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [0.5, 0.3],
            [0.7, 0.4],
        ],
        dtype=torch.double,
    ).unsqueeze(0)
    stream = log_signature(path, depth=2, stream=True)
    full = log_signature(path, depth=2)
    assert torch.allclose(stream[:, -1], full, atol=1e-6)
    for step in range(stream.shape[1]):
        prefix = log_signature(path[:, : step + 2], depth=2)
        assert torch.allclose(stream[:, step], prefix, atol=1e-6)
