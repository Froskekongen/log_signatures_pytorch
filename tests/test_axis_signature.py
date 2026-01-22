import pytest
import torch
from torch import Tensor

from log_signatures_pytorch.axis_signature import extract_sparse_events, signature_axis
from log_signatures_pytorch.signature import signature


def expand_axis_path(
    path: Tensor,
    gmax: int,
    eps: float = 0.0,
    selection: str = "topk",
    order: str = "sorted",
) -> Tensor:
    """
    Helper to expand a path into its full axis-aligned microsteps.
    Returns an expanded path of shape (B, 1 + T * gmax, d).
    """
    B, L, d = path.shape
    T = L - 1
    inc = path[:, 1:] - path[:, :-1]

    # Use the same logic as implementation to get microsteps
    idx, delta, _ = extract_sparse_events(
        inc, gmax, eps, selection, order, check_exact=False
    )

    # idx: (B, T, gmax)
    # delta: (B, T, gmax)

    # Construct micro-increments (B, T, gmax, d)
    micro_incs = torch.zeros(B, T, gmax, d, dtype=path.dtype, device=path.device)

    # We need to scatter delta into the correct coordinate at each microstep
    # dim 3 is 'd'.
    # We want micro_incs[b, t, k, idx[b,t,k]] = delta[b,t,k]

    # Flatten B, T, k to a single batch dim for scatter
    flat_idx = idx.reshape(-1)  # (B*T*gmax)
    flat_delta = delta.reshape(-1)  # (B*T*gmax)
    flat_micro = micro_incs.reshape(-1, d)  # (B*T*gmax, d)

    # scatter
    flat_micro.scatter_(1, flat_idx.unsqueeze(1), flat_delta.unsqueeze(1))

    # Reshape back to (B, T*gmax, d)
    # The sequence of increments is just the flattened microsteps
    seq_incs = micro_incs.reshape(B, T * gmax, d)

    # Reconstruct path
    start_point = path[:, :1]  # (B, 1, d)

    # Check if we need to filter out zero-increments?
    # Technically, 0 increments don't change signature, so keeping them is fine and easier.

    expanded_path = torch.cat(
        [start_point, start_point + torch.cumsum(seq_incs, dim=1)], dim=1
    )
    return expanded_path


class TestAxisSignature:
    def test_depth_2_correctness_vs_expanded(self):
        """Test 1: Depth=2 matches expanded axis-path reference."""
        torch.manual_seed(42)
        B, L, d = 2, 6, 8
        path = torch.randn(B, L, d)

        # Enforce sparsity to be exact
        # Make only 2 coords change per step
        inc = path[:, 1:] - path[:, :-1]
        mask = torch.zeros_like(inc)
        for b in range(B):
            for t in range(L - 1):
                # pick 2 random indices
                indices = torch.randperm(d)[:2]
                mask[b, t, indices] = 1.0

        inc = inc * mask
        # reconstruct path
        path = torch.cat([path[:, :1], path[:, :1] + torch.cumsum(inc, dim=1)], dim=1)

        gmax = 2

        # 1. Compute using sparse backend
        sig_axis = signature_axis(path, depth=2, gmax=gmax, eps=0.0, selection="topk")

        # 2. Compute reference on expanded path
        expanded_path = expand_axis_path(path, gmax=gmax, eps=0.0, selection="topk")
        sig_ref = signature(expanded_path, depth=2)

        torch.testing.assert_close(sig_axis, sig_ref, atol=1e-5, rtol=1e-5)

    def test_depth_3_correctness_vs_expanded(self):
        """Test 2: Depth=3 matches expanded axis-path reference (small d)."""
        torch.manual_seed(42)
        B, L, d = 2, 4, 5
        gmax = 2

        # Create explicit sparse path
        path = torch.randn(B, L, d)
        inc = path[:, 1:] - path[:, :-1]
        mask = torch.zeros_like(inc)
        for b in range(B):
            for t in range(L - 1):
                indices = torch.randperm(d)[:gmax]
                mask[b, t, indices] = 1.0
        inc = inc * mask
        path = torch.cat([path[:, :1], path[:, :1] + torch.cumsum(inc, dim=1)], dim=1)

        sig_axis = signature_axis(path, depth=3, gmax=gmax, selection="topk")

        expanded_path = expand_axis_path(path, gmax=gmax, selection="topk")
        sig_ref = signature(expanded_path, depth=3)

        torch.testing.assert_close(sig_axis, sig_ref, atol=1e-5, rtol=1e-5)

    def test_stream_vs_batch(self):
        """Test 3: stream=True last slice equals stream=False."""
        torch.manual_seed(42)
        B, L, d = 2, 5, 4
        path = torch.randn(B, L, d)
        depth = 2

        # Use gmax=d for dense random data
        sig_stream = signature_axis(path, depth=depth, stream=True, gmax=d)
        sig_final = signature_axis(path, depth=depth, stream=False, gmax=d)

        # sig_stream is (B, L-1, dim)
        # sig_final is (B, dim)
        torch.testing.assert_close(sig_stream[:, -1, :], sig_final)

    def test_g1_equivalence_linear(self):
        """Test 4: g=1 case matches original linear signature."""
        # If only 1 coordinate changes, axis path is same as straight line.
        torch.manual_seed(42)
        B, L, d = 2, 5, 4

        # Create 1-sparse increments
        inc = torch.zeros(B, L - 1, d)
        for b in range(B):
            for t in range(L - 1):
                idx = torch.randint(0, d, (1,))
                inc[b, t, idx] = torch.randn(1)

        path = torch.zeros(B, L, d)
        path[:, 1:] = torch.cumsum(inc, dim=1)

        sig_axis = signature_axis(path, depth=2, gmax=1)
        sig_linear = signature(path, depth=2)

        torch.testing.assert_close(sig_axis, sig_linear, atol=1e-5, rtol=1e-5)

    def test_order_sensitivity(self):
        """Test 5: Order sensitivity sanity check."""
        # Single step, d=2, both change.
        # Path: (0,0) -> (1,1)
        # Sorted (0 then 1): (0,0)->(1,0)->(1,1)
        # Reverse (1 then 0): (0,0)->(0,1)->(1,1)
        # Area (Levy area) should differ.

        path = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])  # (1, 2, 2)

        sig_sorted = signature_axis(path, depth=2, order="sorted", gmax=2)
        sig_reverse = signature_axis(path, depth=2, order="reverse", gmax=2)

        # Level 1 should be same (total displacement)
        d = 2
        dim1 = d
        torch.testing.assert_close(sig_sorted[:, :dim1], sig_reverse[:, :dim1])

        # Level 2 should differ
        assert not torch.allclose(sig_sorted[:, dim1:], sig_reverse[:, dim1:])

    def test_exactness_guard(self):
        """Test 6: Exactness guard raises error."""
        # Make all coordinates change
        B, L, d = 1, 2, 4
        path = torch.randn(B, L, d)
        gmax = 2  # less than d=4

        with pytest.raises(RuntimeError, match="Axis signature precision loss"):
            signature_axis(path, depth=2, gmax=gmax, check_exact=True, eps=0.0)

    def test_device_smoke(self):
        """Test 7: Smoke test for device/dtype."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        path = torch.randn(2, 5, 4, device="cuda", dtype=torch.float32)
        sig = signature_axis(path, depth=2, gmax=4)
        assert sig.device.type == "cuda"
        assert sig.dtype == torch.float32

    def test_grad_flow(self):
        """Test 8: Gradients propagate through the signature."""
        path = torch.randn(1, 4, 3, requires_grad=True)
        sig = signature_axis(path, depth=2, gmax=3)
        loss = sig.sum()
        loss.backward()
        assert path.grad is not None
        assert not torch.isnan(path.grad).any()
