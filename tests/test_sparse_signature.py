"""Tests for sparse signature computation."""

from typing import cast

import pytest
import torch

from log_signatures_pytorch.signature import signature
from log_signatures_pytorch.sparse_signature import (
    _knot_indices_from_repeats,
    _sparse_increments,
    pad_paths_correctly,
    signature_sparse,
)


def test_knot_indices_basic() -> None:
    """Test knot extraction for a simple path with repeats."""
    path = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]
    )  # (batch=1, T=5, D=2)
    knots = _knot_indices_from_repeats(path)
    # Should include 0, 2 (first change), 4 (last point)
    assert knots.shape[0] == 1
    assert knots[0, 0] == 0
    assert knots[0, 1] == 2
    assert knots[0, 2] == 4


def test_knot_indices_all_identical() -> None:
    """Test knot extraction when all points are identical."""
    path = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    knots = _knot_indices_from_repeats(path)
    # Should include 0 and last point (2)
    assert knots[0, 0] == 0
    assert knots[0, 1] == 2


def test_knot_indices_with_eps() -> None:
    """Test knot extraction with epsilon threshold."""
    path = torch.tensor(
        [[[0.0, 0.0], [0.001, 0.001], [1.0, 1.0], [1.001, 1.001], [2.0, 0.0]]]
    )
    # With eps=0.01, small changes should be ignored
    knots = _knot_indices_from_repeats(path, eps=0.01)
    # Should skip the small changes
    assert knots[0, 0] == 0
    # Next knot should be at index where change > 0.01
    assert knots[0, 1] == 2  # First significant change
    assert knots[0, 2] == 4  # Last point


def test_sparse_increments_basic() -> None:
    """Test sparse increment extraction."""
    path = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]])
    increments, counts = _sparse_increments(path)
    assert counts[0] == 3  # 3 knots means 2 segments
    assert increments.shape == (1, 2, 2)
    # First increment: [0,0] -> [1,1] = [1,1]
    torch.testing.assert_close(increments[0, 0], torch.tensor([1.0, 1.0]))
    # Second increment: [1,1] -> [2,0] = [1,-1]
    torch.testing.assert_close(increments[0, 1], torch.tensor([1.0, -1.0]))


def test_signature_sparse_vs_naive_basic() -> None:
    """Test that sparse signature matches naive signature for a simple path."""
    # Simple path without repeats
    path = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]])
    depth = 3

    sig_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth))
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_with_repeats() -> None:
    """Test sparse signature with repeated points."""
    # Path with many repeats
    path = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]
    )
    depth = 3

    sig_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth))
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_all_identical() -> None:
    """Test sparse signature when all points are identical."""
    path = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    depth = 3

    sig_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth))
    # Signature of constant path should be identity: S0=1, all higher levels zero
    # Flattened: [1, 0, 0, 0, ...] but signature() returns levels 1..depth
    sig_naive = signature(path, depth=depth)

    # Both should be zero (no increments)
    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)
    # Should be all zeros (no path increments)
    assert torch.allclose(sig_sparse, torch.zeros_like(sig_sparse), atol=1e-6)


def test_signature_sparse_single_step() -> None:
    """Test sparse signature for T=2 (single step)."""
    path = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    depth = 2

    sig_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth))
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_batch() -> None:
    """Test sparse signature with batched paths."""
    path = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],  # Path 1
            [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]],  # Path 2
        ]
    )
    depth = 2

    sig_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth))
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_signature_sparse_with_lengths() -> None:
    """Test sparse signature with padded batch and lengths."""
    path = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 0.0]],  # Valid length 3
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],  # Valid length 3
        ]
    )
    lengths = torch.tensor([3, 3])
    depth = 2

    sig_sparse = cast(
        torch.Tensor, signature_sparse(path, depth=depth, lengths=lengths)
    )
    sig_naive = signature(path[:, :3], depth=depth)  # Compare with unpadded

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-6, rtol=1e-6)


def test_pad_paths_correctly_repeats_last_point() -> None:
    """Correct padding (repeat last point) should be signature-safe."""
    torch.manual_seed(0)
    depth = 3

    paths = [
        torch.randn(5, 2, dtype=torch.float64).cumsum(dim=0),
        torch.randn(3, 2, dtype=torch.float64).cumsum(dim=0),
        torch.randn(8, 2, dtype=torch.float64).cumsum(dim=0),
    ]
    padded, lengths = pad_paths_correctly(paths)

    # Padding repeats last point
    for b, p in enumerate(paths):
        t = p.shape[0]
        if t < padded.shape[1]:
            torch.testing.assert_close(
                padded[b, t:], p[-1].expand(padded.shape[1] - t, -1)
            )

    # Sparse on padded batch (no lengths) should match per-path unpadded signatures
    sig_sparse = cast(torch.Tensor, signature_sparse(padded, depth=depth))
    sig_ref = torch.cat([signature(p.unsqueeze(0), depth=depth) for p in paths], dim=0)
    torch.testing.assert_close(sig_sparse, sig_ref, atol=1e-6, rtol=1e-6)

    # lengths returned are correct
    torch.testing.assert_close(lengths.cpu(), torch.tensor([5, 3, 8]))


def test_signature_sparse_return_levels() -> None:
    """Test sparse signature with return_levels=True."""
    path = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]])
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

    sig = cast(torch.Tensor, signature_sparse(path, depth=depth))
    loss = sig.sum()
    loss.backward()

    assert path.grad is not None
    assert torch.isfinite(path.grad).all()


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_signature_sparse_vs_naive_random(depth: int) -> None:
    """Test sparse vs naive on random variable-length paths with repeats."""
    torch.manual_seed(123)
    batch_size = 2
    base_length = 100
    width = 3

    # Create random walk
    increments = torch.randn(batch_size, base_length, width, dtype=torch.float64)
    path_base = torch.cumsum(
        torch.cat([torch.zeros(batch_size, 1, width), increments], dim=1), dim=1
    )

    # Inject repeats: randomly repeat some points
    path_list = []
    for b in range(batch_size):
        path_b = [path_base[b, 0]]
        for i in range(1, base_length):
            path_b.append(path_base[b, i])
            # Randomly repeat the point 0-3 times
            num_repeats = int(torch.randint(0, 4, (1,)).item())
            for _ in range(num_repeats):
                path_b.append(path_base[b, i])
        path_list.append(torch.stack(path_b))

    padded, _lengths = pad_paths_correctly(path_list)

    # Compare sparse on padded batch to per-path unpadded signatures (correctness)
    sig_sparse = cast(torch.Tensor, signature_sparse(padded, depth=depth))
    sig_ref = torch.cat(
        [signature(p.unsqueeze(0), depth=depth) for p in path_list], dim=0
    )

    torch.testing.assert_close(sig_sparse, sig_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_signature_sparse_cuda() -> None:
    """Test sparse signature on CUDA."""
    device = torch.device("cuda")
    path = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]], device=device
    )
    depth = 2

    sig_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth))
    sig_naive = signature(path, depth=depth)

    torch.testing.assert_close(sig_sparse, sig_naive, atol=1e-5, rtol=1e-5)


def test_signature_sparse_single_path() -> None:
    """Test sparse signature with single path (batched with batch_size=1)."""
    path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(
        0
    )  # (1, T, D) - single path in batch
    depth = 2

    sig = cast(torch.Tensor, signature_sparse(path, depth=depth))
    # Should return 2D tensor (batch, sig_dim)
    assert sig.ndim == 2
    assert sig.shape[0] == 1


def test_sparse_signature_stream_exact_match() -> None:
    """Test that sparse signature stream matches dense signature stream."""
    # Create a path with repeated points
    # A -> A -> B -> B -> C
    path = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]])
    depth = 2

    # Dense stream
    dense_stream = signature(path, depth=depth, stream=True)

    # Sparse stream
    sparse_stream = cast(torch.Tensor, signature_sparse(path, depth=depth, stream=True))

    assert sparse_stream.shape == dense_stream.shape
    torch.testing.assert_close(sparse_stream, dense_stream, atol=1e-6, rtol=1e-6)


def test_sparse_signature_stream_with_epsilon() -> None:
    """Test sparse stream with epsilon > 0."""
    # Path with small noise that should be filtered
    # A -> A' -> B where A' is close to A
    path = torch.tensor([[[0.0, 0.0], [0.01, 0.01], [1.0, 1.0]]])
    depth = 2
    eps = 0.02

    # With eps=0.02, [0.01, 0.01] is within distance of [0.0, 0.0]
    # So it should be treated as repeat of A.
    # Knots: 0, 2 (index 1 is skipped).
    # Increment: path[2] - path[0] = [1.0, 1.0].

    stream = cast(
        torch.Tensor, signature_sparse(path, depth=depth, eps=eps, stream=True)
    )

    # Length of path is 3. Stream length is 2.
    # t=1: A -> A' (skipped, so Identity)
    # t=2: A -> B (jump to B)

    assert stream.shape == (1, 2, 6)  # width=2, depth=2 -> 2+4=6 dims

    # First point should be zero (Identity in flattened form implies zeros)
    torch.testing.assert_close(stream[0, 0], torch.zeros_like(stream[0, 0]))

    # Second point should be signature of segment [0.0, 0.0] -> [1.0, 1.0]
    expected_sig = signature(torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]), depth=depth)[0]
    torch.testing.assert_close(stream[0, 1], expected_sig)


def test_sparse_signature_stream_return_levels() -> None:
    """Test return_levels=True with streaming."""
    path = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]])
    depth = 2

    levels = signature_sparse(path, depth=depth, stream=True, return_levels=True)

    assert isinstance(levels, list)
    assert len(levels) == depth
    # Level 1: (batch, T-1, dim)
    assert levels[0].shape == (1, 2, 2)
    # Level 2: (batch, T-1, dim, dim)
    assert levels[1].shape == (1, 2, 2, 2)

    # Check values
    # t=1: Zero (Identity)
    torch.testing.assert_close(levels[0][0, 0], torch.zeros(2))
    # t=2: [1.0, 1.0]
    torch.testing.assert_close(levels[0][0, 1], torch.tensor([1.0, 1.0]))


def test_sparse_signature_stream_batch() -> None:
    """Test batched sparse streaming."""
    path = torch.tensor([[[0.0], [0.0], [1.0]], [[0.0], [1.0], [1.0]]])  # (2, 3, 1)

    # Path 0: A A B. Knots 0, 2. Stream: Identity, Sig(B-A).
    # Path 1: A B B. Knots 0, 1, 2. Stream: Sig(B-A), Sig(B-A).

    res = cast(torch.Tensor, signature_sparse(path, depth=1, stream=True))

    # Path 0
    torch.testing.assert_close(res[0, 0], torch.zeros(1))
    torch.testing.assert_close(res[0, 1], torch.tensor([1.0]))

    # Path 1
    # t=1: A->B. Sig 1.0.
    torch.testing.assert_close(res[1, 0], torch.tensor([1.0]))
    # t=2: B->B. Sig should remain 1.0 (since increment is 0, S tensor 1 = S).
    torch.testing.assert_close(res[1, 1], torch.tensor([1.0]))


def test_sparse_signature_stream_long_random() -> None:
    """Test streaming signature on a long path with random repeats."""
    torch.manual_seed(999)
    batch_size = 2
    base_length = 50
    width = 3
    depth = 3

    # Create random walk (unique points)
    increments = torch.randn(batch_size, base_length, width, dtype=torch.float64)
    # Start at 0
    path_base = torch.cumsum(
        torch.cat(
            [torch.zeros(batch_size, 1, width, dtype=torch.float64), increments], dim=1
        ),
        dim=1,
    )

    # Inject repeats randomly
    path_list = []
    for b in range(batch_size):
        new_path = []
        for i in range(path_base.shape[1]):
            val = path_base[b, i]
            new_path.append(val)
            # Repeat 0 to 5 times
            repeats = int(torch.randint(0, 6, (1,)).item())
            for _ in range(repeats):
                new_path.append(val)
        path_list.append(torch.stack(new_path))

    # Pad
    max_len = max(len(p) for p in path_list)
    path = torch.zeros(batch_size, max_len, width, dtype=torch.float64)
    # Fill with last value for padding (effectively repeating the last point)
    for b, p in enumerate(path_list):
        path[b, : len(p)] = p
        path[b, len(p) :] = p[-1]

    # Compute both
    stream_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth, stream=True))
    stream_dense = signature(path, depth=depth, stream=True)

    assert stream_sparse.shape == stream_dense.shape
    # Use slightly higher tolerance due to cumulative float ops difference between
    # compressed (fewer ops) and dense (more ops adding zeros).
    # But usually adding zeros shouldn't drift much.
    torch.testing.assert_close(stream_sparse, stream_dense, atol=1e-6, rtol=1e-6)


def test_sparse_signature_stream_all_identical_long() -> None:
    """Test streaming signature on a long path of identical points."""
    batch_size = 2
    length = 50
    width = 3
    depth = 2

    path = torch.zeros(batch_size, length, width)
    # Add some offset to make it non-zero values, but still identical
    path = path + torch.randn(batch_size, 1, width)

    stream_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth, stream=True))
    stream_dense = signature(path, depth=depth, stream=True)

    # All increments are zero, so signature stream should be all zeros (identity - 1 if we considered the '1' term)
    # The signature function returns terms starting from level 1, so pure zeros.
    assert torch.allclose(stream_sparse, torch.zeros_like(stream_sparse))
    torch.testing.assert_close(stream_sparse, stream_dense)


def test_sparse_signature_stream_no_repeats_long() -> None:
    """Test streaming signature on a long path with NO repeats."""
    torch.manual_seed(42)
    batch_size = 1
    length = 50
    width = 2
    depth = 2

    # Ensure no repeats by adding non-zero noise everywhere
    path = torch.cumsum(
        torch.randn(batch_size, length, width, dtype=torch.float64), dim=1
    )

    stream_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth, stream=True))
    stream_dense = signature(path, depth=depth, stream=True)

    torch.testing.assert_close(stream_sparse, stream_dense, atol=1e-6, rtol=1e-6)


def test_sparse_signature_stream_alternating() -> None:
    """Test streaming signature on alternating path A -> B -> A -> B ..."""
    # This creates a lot of knots but also repeats if we structure it right,
    # or just tests that the knot logic works when every other point is a knot.
    # A -> A -> B -> B -> A -> A ...
    depth = 2
    A = torch.tensor([0.0, 0.0])
    B = torch.tensor([1.0, 1.0])

    # Construct path: A, A, B, B, A, A, B, B
    points = []
    for _ in range(10):
        points.extend([A, A, B, B])
    path = torch.stack(points).unsqueeze(0)  # (1, 40, 2)

    stream_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth, stream=True))
    stream_dense = signature(path, depth=depth, stream=True)

    torch.testing.assert_close(stream_sparse, stream_dense, atol=1e-6, rtol=1e-6)


def test_sparse_signature_stream_padding_handling() -> None:
    """Test that streaming handles padded batches correctly."""
    # Two paths of different effective lengths
    # Path 1: A->B->C (Length 3)
    # Path 2: A->B->C->D->E (Length 5)
    # Both padded to Length 6 (with repeats of last element usually for padding in this context)
    # Or just garbage padding?
    # If we use `lengths` argument, we can ignore garbage.

    # Let's use repeats as padding since that's natural for signatures (zero increment)
    path = torch.tensor(
        [
            [
                [0.0],
                [1.0],
                [2.0],
                [2.0],
                [2.0],
            ],  # Effective len 3 (indices 0,1,2). Padded with 2.0.
            [[0.0], [1.0], [2.0], [3.0], [4.0]],  # Effective len 5.
        ]
    )
    depth = 2
    # We expect the signature of Path 1 to freeze after index 2.
    # T=0 (val=0) -> T=1 (val=1): Sig change
    # T=1 (val=1) -> T=2 (val=2): Sig change
    # T=2 (val=2) -> T=3 (val=2): No change
    # ...

    stream_sparse = cast(torch.Tensor, signature_sparse(path, depth=depth, stream=True))
    stream_dense = signature(path, depth=depth, stream=True)

    torch.testing.assert_close(stream_sparse, stream_dense, atol=1e-6, rtol=1e-6)

    # Check specifically that Path 1 stays constant at the end
    # Output shape: (2, 4, dim) corresponding to steps 0->1, 1->2, 2->3, 3->4

    # So Index 2 should equal Index 1.
    # Index 3 should equal Index 2.
    torch.testing.assert_close(stream_sparse[0, 2], stream_sparse[0, 1])
    torch.testing.assert_close(stream_sparse[0, 3], stream_sparse[0, 2])
