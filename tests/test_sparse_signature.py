"""Tests for sparse signature computation."""

import pytest
import torch

from log_signatures_pytorch.signature import signature
from log_signatures_pytorch.sparse_signature import (
    knot_indices_from_repeats,
    signature_sparse,
    sparse_increments,
)


def test_knot_indices_basic() -> None:
    """Test knot extraction for a simple path with repeats."""
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ])  # (batch=1, T=5, D=2)
    knots = knot_indices_from_repeats(path)
    # Should include 0, 2 (first change), 4 (last point)
    assert knots.shape[0] == 1
    assert knots[0, 0] == 0
    assert knots[0, 1] == 2
    assert knots[0, 2] == 4


def test_knot_indices_all_identical() -> None:
    """Test knot extraction when all points are identical."""
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    ])
    knots = knot_indices_from_repeats(path)
    # Should include 0 and last point (2)
    assert knots[0, 0] == 0
    assert knots[0, 1] == 2


def test_knot_indices_with_eps() -> None:
    """Test knot extraction with epsilon threshold."""
    path = torch.tensor([
        [[0.0, 0.0], [0.001, 0.001], [1.0, 1.0], [1.001, 1.001], [2.0, 0.0]]
    ])
    # With eps=0.01, small changes should be ignored
    knots = knot_indices_from_repeats(path, eps=0.01)
    # Should skip the small changes
    assert knots[0, 0] == 0
    # Next knot should be at index where change > 0.01
    assert knots[0, 1] == 2  # First significant change
    assert knots[0, 2] == 4  # Last point


def test_sparse_increments_basic() -> None:
    """Test sparse increment extraction."""
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ])
    increments, counts = sparse_increments(path)
    assert counts[0] == 3  # 3 knots means 2 segments
    assert increments.shape == (1, 2, 2)
    # First increment: [0,0] -> [1,1] = [1,1]
    torch.testing.assert_close(increments[0, 0], torch.tensor([1.0, 1.0]))
    # Second increment: [1,1] -> [2,0] = [1,-1]
    torch.testing.assert_close(increments[0, 1], torch.tensor([1.0, -1.0]))


def test_signature_sparse_vs_naive_basic() -> None:
    """Test that sparse signature matches naive signature for a simple path."""
    # Simple path without repeats
    path = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    ])
    depth = 3

    sig_sparse = signature_sparse(path, depth=depth)
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_with_repeats() -> None:
    """Test sparse signature with repeated points."""
    # Path with many repeats
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    ])
    depth = 3

    sig_sparse = signature_sparse(path, depth=depth)
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_all_identical() -> None:
    """Test sparse signature when all points are identical."""
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    ])
    depth = 3

    sig_sparse = signature_sparse(path, depth=depth)
    # Signature of constant path should be identity: S0=1, all higher levels zero
    # Flattened: [1, 0, 0, 0, ...] but signature() returns levels 1..depth
    sig_naive = signature(path, depth=depth)

    # Both should be zero (no increments)
    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)
    # Should be all zeros (no path increments)
    assert torch.allclose(sig_sparse, torch.zeros_like(sig_sparse), atol=1e-6)


def test_signature_sparse_single_step() -> None:
    """Test sparse signature for T=2 (single step)."""
    path = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0]]
    ])
    depth = 2

    sig_sparse = signature_sparse(path, depth=depth)
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_batch() -> None:
    """Test sparse signature with batched paths."""
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],  # Path 1
        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]],  # Path 2
    ])
    depth = 2

    sig_sparse = signature_sparse(path, depth=depth)
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_with_lengths() -> None:
    """Test sparse signature with padded batch and lengths."""
    path = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]],  # Valid length 3
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],  # Valid length 3
    ])
    lengths = torch.tensor([3, 3])
    depth = 2

    sig_sparse = signature_sparse(path, depth=depth, lengths=lengths)
    sig_naive = signature(path[:, :3], depth=depth)  # Compare with unpadded

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_return_levels() -> None:
    """Test sparse signature with return_levels=True."""
    path = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    ])
    depth = 2

    levels = signature_sparse(path, depth=depth, return_levels=True)
    assert isinstance(levels, list)
    assert len(levels) == depth  # Levels 1..depth (excluding E_0)

    # Check shapes
    assert levels[0].shape == (1, 2)  # Level 1: (batch, width)
    assert levels[1].shape == (1, 2, 2)  # Level 2: (batch, width, width)


def test_signature_sparse_grad_propagates() -> None:
    """Test that gradients flow through sparse signature."""
    torch.manual_seed(42)
    path = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    depth = 3

    sig = signature_sparse(path, depth=depth)
    loss = sig.sum()
    loss.backward()

    assert path.grad is not None
    assert torch.isfinite(path.grad).all()


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_signature_sparse_vs_naive_random(depth: int) -> None:
    """Test sparse vs naive on random paths with injected repeats."""
    torch.manual_seed(123)
    batch_size = 2
    base_length = 100
    width = 3

    # Create random walk
    increments = torch.randn(batch_size, base_length, width, dtype=torch.float64)
    path_base = torch.cumsum(torch.cat([torch.zeros(batch_size, 1, width), increments], dim=1), dim=1)

    # Inject repeats: randomly repeat some points
    path_list = []
    for b in range(batch_size):
        path_b = [path_base[b, 0]]
        for i in range(1, base_length):
            path_b.append(path_base[b, i])
            # Randomly repeat the point 0-3 times
            num_repeats = torch.randint(0, 4, (1,)).item()
            for _ in range(num_repeats):
                path_b.append(path_base[b, i])
        path_list.append(torch.stack(path_b))

    # Pad to same length
    max_len = max(len(p) for p in path_list)
    path = torch.zeros(batch_size, max_len, width, dtype=torch.float64)
    for b, p in enumerate(path_list):
        path[b, :len(p)] = p

    sig_sparse = signature_sparse(path, depth=depth)
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_signature_sparse_cuda() -> None:
    """Test sparse signature on CUDA."""
    device = torch.device("cuda")
    path = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    ], device=device)
    depth = 2

    sig_sparse = signature_sparse(path, depth=depth)
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-5, rtol=1e-5)


def test_signature_sparse_single_path() -> None:
    """Test sparse signature with single path (2D input)."""
    path = torch.tensor([
        [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]
    ])  # (T, D) - no batch dimension
    depth = 2

    sig = signature_sparse(path, depth=depth)
    # Should return 1D tensor (single path)
    assert sig.ndim == 1

    # Compare with batched version
    path_batched = path.unsqueeze(0)
    sig_batched = signature_sparse(path_batched, depth=depth)
    torch.testing.assert_close(sig, sig_batched[0], atol=1e-6, rtol=1e-6)
