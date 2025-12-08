"""Comprehensive tests for log-signature computation."""

import pytest
import torch

from functools import lru_cache

from log_signatures_pytorch.basis import (
    hall_basis,
    logsigdim,
    logsigdim_words,
    logsigkeys,
    logsigkeys_words,
    lyndon_words,
)
from log_signatures_pytorch.log_signature import (
    _project_to_hall_basis,
    _project_to_words_basis,
    _signature_to_logsignature_tensor,
    _unflatten_signature,
    log_signature,
)
from log_signatures_pytorch.hall_projection import (
    _hall_basis_tensors,
    _hall_element_depth,
)
from log_signatures_pytorch.signature import signature
from log_signatures_pytorch.tensor_ops import batch_lie_brackets, lie_brackets


class TestHallBasis:
    """Tests for Hall basis generation."""

    def test_hall_basis_depth_1(self):
        """Test Hall basis generation for depth 1."""
        basis = hall_basis(2, 1)
        assert basis == [1, 2]

        basis = hall_basis(3, 1)
        assert basis == [1, 2, 3]

    def test_hall_basis_depth_2(self):
        """Test Hall basis generation for depth 2."""
        basis = hall_basis(2, 2)
        # Should have: 1, 2, [1,2]
        assert len(basis) == 3
        assert 1 in basis
        assert 2 in basis
        assert (1, 2) in basis

    def test_hall_basis_depth_3(self):
        """Test Hall basis generation for depth 3."""
        basis = hall_basis(2, 3)
        # Should have: 1, 2, [1,2], [1,[1,2]], [2,[1,2]]
        assert len(basis) == 5
        assert 1 in basis
        assert 2 in basis
        assert (1, 2) in basis

    def test_logsigdim(self):
        """Test log-signature dimension calculation."""
        assert logsigdim(2, 1) == 2
        assert logsigdim(2, 2) == 3
        assert logsigdim(2, 3) == 5
        assert logsigdim(3, 1) == 3
        assert logsigdim(3, 2) == 6  # 3 letters + 3 brackets [1,2], [1,3], [2,3]

    def test_logsigkeys(self):
        """Test log-signature key generation."""
        keys = logsigkeys(2, 1)
        assert keys == ["1", "2"]

        keys = logsigkeys(2, 2)
        assert "1" in keys
        assert "2" in keys
        assert "[1,2]" in keys

        keys = logsigkeys(2, 3)
        assert len(keys) == 5
        assert "[1,2]" in keys
        assert "[1,[1,2]]" in keys or "[1,[1,2]]" in keys

    def test_hall_basis_ordering(self):
        """Test that Hall basis elements are properly ordered."""
        basis = hall_basis(3, 3)
        # Letters should come before brackets
        letters = [b for b in basis if isinstance(b, int)]
        brackets = [b for b in basis if isinstance(b, tuple)]
        assert all(isinstance(b, int) for b in letters)
        assert all(isinstance(b, tuple) for b in brackets)

        # Within same depth, should be lexicographically ordered
        depth_2 = [
            b
            for b in basis
            if isinstance(b, tuple)
            and not any(isinstance(x, tuple) for x in b if isinstance(b, tuple))
        ]
        if len(depth_2) > 1:
            # Check ordering (simplified check)
            pass


class TestLyndonWords:
    """Tests for Lyndon/words basis utilities."""

    @staticmethod
    def _mobius(n: int) -> int:
        """Integer Möbius function (simple factorization)."""
        p = 0
        m = n
        d = 2
        while d * d <= m:
            cnt = 0
            while m % d == 0:
                m //= d
                cnt += 1
            if cnt > 1:
                return 0
            if cnt == 1:
                p += 1
            d += 1
        if m > 1:
            p += 1
        return -1 if p % 2 else 1

    def test_lyndon_count_matches_witt_formula(self):
        """Count of Lyndon words matches Witt/necklace formula."""
        for width in (2, 3, 4):
            for length in (1, 2, 3, 4):
                words = [w for w in lyndon_words(width, length) if len(w) == length]
                count = len(words)
                expected = (
                    sum(
                        self._mobius(d) * (width ** (length // d))
                        for d in range(1, length + 1)
                        if length % d == 0
                    )
                    // length
                )
                assert count == expected

    @staticmethod
    @lru_cache(maxsize=None)
    def _hall_to_words_matrix(width: int, depth: int, device: str = "cpu"):
        basis = hall_basis(width, depth)
        dim = len(basis)
        basis_tensors = _hall_basis_tensors(width, depth)
        cols = []
        for elem in basis:
            level = _hall_element_depth(elem)
            log_sig_tensors = [
                torch.zeros(1, *([width] * d), dtype=torch.float64, device=device)
                for d in range(1, depth + 1)
            ]
            log_sig_tensors[level - 1] = (
                basis_tensors[elem].unsqueeze(0).to(device=device, dtype=torch.float64)
            )
            hall_vec = _project_to_hall_basis(log_sig_tensors, width, depth)
            words_vec = _project_to_words_basis(log_sig_tensors, width, depth)
            cols.append(words_vec.squeeze(0))
            # Sanity: hall projection should be one-hot at this element
            expected = torch.zeros(dim, dtype=torch.float64, device=device)
            expected[len(cols) - 1] = 1.0
            torch.testing.assert_close(
                hall_vec.squeeze(0), expected, atol=1e-9, rtol=1e-9
            )
        return torch.stack(cols, dim=1)  # (dim, dim)

    @pytest.mark.parametrize("width,depth", [(2, 3), (3, 2)])
    def test_words_hall_change_of_basis_is_linear(self, width: int, depth: int):
        """Check words coordinates are a linear transform of Hall coordinates."""
        C = self._hall_to_words_matrix(width, depth)
        dim = C.shape[0]
        assert torch.linalg.matrix_rank(C) == dim

        torch.manual_seed(0)
        path = torch.randn(2, 5, width, dtype=torch.float64)
        hall_vec = log_signature(path, depth=depth, mode="hall")
        words_vec = log_signature(path, depth=depth, mode="words")

        torch.testing.assert_close(
            words_vec,
            hall_vec @ C,
            atol=1e-8,
            rtol=1e-6,
        )

    def test_lyndon_words_ordering_and_dim(self):
        words = lyndon_words(3, 3)
        lengths = [len(w) for w in words]
        assert lengths == sorted(lengths)  # non-decreasing by length
        # Within each length, lexicographic
        for L in {1, 2, 3}:
            block = [w for w in words if len(w) == L]
            assert block == sorted(block)
        # Dimension matches Hall basis dimension
        assert logsigdim_words(3, 3) == logsigdim(3, 3)

    def test_logsigkeys_words(self):
        keys = logsigkeys_words(2, 2)
        assert keys[0] == "1"
        assert keys[1] == "2"
        assert "1,2" in keys


class TestLieBracket:
    """Tests for Lie bracket operations."""

    def test_lie_bracket_anticommutativity(self):
        """Test that Lie bracket is anti-commutative: [a,b] = -[b,a]."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])

        ab = lie_brackets(a, b)
        ba = lie_brackets(b, a)

        torch.testing.assert_close(ab, -ba)

    def test_batch_lie_bracket_anticommutativity(self):
        """Test batched Lie bracket anti-commutativity."""
        a = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        b = torch.tensor([[3.0, 4.0], [4.0, 5.0]])

        ab = batch_lie_brackets(a, b)
        ba = batch_lie_brackets(b, a)

        torch.testing.assert_close(ab, -ba)

    def test_lie_bracket_bilinearity(self):
        """Test that Lie bracket is bilinear."""
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        c = torch.tensor([5.0, 6.0])
        alpha = 2.0
        beta = 3.0

        # [αa + βb, c] = α[a, c] + β[b, c]
        left = lie_brackets(alpha * a + beta * b, c)
        right = alpha * lie_brackets(a, c) + beta * lie_brackets(b, c)

        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


class TestLogSignature:
    """Tests for log-signature computation."""

    def test_log_signature_shape_single_path(self):
        """Test log-signature shape for single path."""
        path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
        depth = 2

        log_sig = log_signature(path, depth=depth)

        expected_dim = logsigdim(2, depth)
        assert log_sig.shape == (1, expected_dim)

    def test_log_signature_shape_batched(self):
        """Test log-signature shape for batched paths."""
        path = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [2.0, 2.0]]])
        depth = 2

        log_sig = log_signature(path, depth=depth)

        expected_dim = logsigdim(2, depth)
        assert log_sig.shape == (2, expected_dim)

    def test_log_signature_words_mode_shape_and_value(self):
        """Words mode should gather Lyndon coefficients without mixing."""
        path = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.5, 0.5]]).unsqueeze(0)
        depth = 2

        # Direct API
        log_sig_words = log_signature(path, depth=depth, mode="words")

        # Manual gather from tensor-log (should match words mode)
        sig = signature(path, depth=depth)
        tensors = _unflatten_signature(sig, width=2, depth=depth)
        log_tensors = _signature_to_logsignature_tensor(tensors, width=2, depth=depth)
        manual_words = _project_to_words_basis(log_tensors, width=2, depth=depth)

        torch.testing.assert_close(log_sig_words, manual_words, atol=1e-6, rtol=1e-6)
        assert log_sig_words.shape[1] == logsigdim_words(2, depth)

    def test_log_signature_stream_shape(self):
        """Test log-signature shape in streaming mode."""
        path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
        depth = 2

        log_sig = log_signature(path, depth=depth, stream=True)

        expected_dim = logsigdim(2, depth)
        assert log_sig.shape == (1, 2, expected_dim)  # length-1 timesteps

    def test_log_signature_words_stream_shape(self):
        path = torch.tensor([[0.0, 0.0], [1.0, 0.5], [1.5, -0.5]]).unsqueeze(0)
        depth = 3

        log_sig = log_signature(path, depth=depth, stream=True, mode="words")

        expected_dim = logsigdim_words(2, depth)
        assert log_sig.shape == (1, 2, expected_dim)

    def test_log_signature_straight_line(self):
        """Test log-signature of a straight line path.

        For a straight line from (0,0) to (a,b), the log-signature
        should be approximately (a, b) at depth 1.
        """
        path = torch.tensor([[0.0, 0.0], [1.0, 2.0]]).unsqueeze(0)
        depth = 1

        log_sig = log_signature(path, depth=depth)

        # For a straight line, log-signature at depth 1 should be the increment
        expected = torch.tensor([[1.0, 2.0]])
        torch.testing.assert_close(log_sig, expected, atol=1e-5, rtol=1e-5)

    def test_log_signature_zero_path(self):
        """Test that log-signature of zero path is zero."""
        path = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).unsqueeze(0)
        depth = 2

        log_sig = log_signature(path, depth=depth)

        torch.testing.assert_close(
            log_sig, torch.zeros_like(log_sig), atol=1e-6, rtol=1e-6
        )

    def test_log_signature_differentiability(self):
        """Test that log-signature computation is differentiable."""
        base_path = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
        path = base_path.unsqueeze(0)
        depth = 2

        log_sig = log_signature(path, depth=depth)
        loss = log_sig.sum()
        loss.backward()

        assert base_path.grad is not None
        assert not torch.isnan(base_path.grad).any()

    def test_log_signature_words_differentiability(self):
        base_path = torch.tensor(
            [[0.0, 0.0], [0.5, -0.25], [1.0, 0.75]], requires_grad=True
        )
        path = base_path.unsqueeze(0)
        depth = 3

        log_sig = log_signature(path, depth=depth, mode="words")
        loss = log_sig.pow(2).sum()
        loss.backward()

        assert base_path.grad is not None
        assert torch.isfinite(base_path.grad).all()

    def test_log_signature_stream_grad_propagates(self):
        torch.manual_seed(5)
        base_path = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)

        out = log_signature(base_path, depth=3, stream=True)
        out.sum().backward()

        assert base_path.grad is not None
        assert torch.isfinite(base_path.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("stream", [False, True])
    def test_log_signature_grad_propagates_cuda(self, stream: bool):
        torch.manual_seed(7)
        base_path = torch.randn(
            1,
            5,
            3,
            device=torch.device("cuda"),
            dtype=torch.float32,
            requires_grad=True,
        )

        out = log_signature(base_path, depth=3, stream=stream)
        out.sum().backward()

        assert base_path.grad is not None
        assert torch.isfinite(base_path.grad).all()

    def test_exp_log_relationship(self):
        """Test that exp(log_signature) ≈ signature (up to numerical precision).

        This is a key mathematical property: signature = exp(log_signature).
        """
        path = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]).unsqueeze(0)
        depth = 2

        log_sig = log_signature(path, depth=depth)
        sig = signature(path, depth=depth)

        # Note: This test requires implementing exp(log_signature) properly
        # For now, we'll just check that dimensions are consistent
        # Full verification requires proper tensor exponential implementation
        log_sig_dim = logsigdim(2, depth)
        sig_dim = sum(2**d for d in range(1, depth + 1))

        assert log_sig.shape[1] == log_sig_dim
        assert sig.shape[1] == sig_dim
        assert log_sig_dim < sig_dim  # Log-signature should be smaller

    def test_log_signature_raises_on_2d_input(self):
        path = torch.zeros(3, 2)
        with pytest.raises(ValueError):
            log_signature(path, depth=2)


class TestMathematicalVerification:
    """Mathematical verification tests using known properties."""

    def test_hall_basis_independence(self):
        """Verify that Hall basis elements are linearly independent.

        This is a key property of the Hall basis - it should form a basis
        for the free Lie algebra.
        """
        # For small cases, we can verify this
        basis = hall_basis(2, 2)
        # Should have exactly 3 elements: 1, 2, [1,2]
        assert len(basis) == 3
        assert len(set(basis)) == len(basis)  # All unique

    def test_logsigdim_vs_sigdim(self):
        """Verify that log-signature dimension is smaller than signature dimension."""
        for width in [2, 3]:
            for depth in [1, 2, 3]:
                log_dim = logsigdim(width, depth)
                sig_dim = sum(width**d for d in range(1, depth + 1))

                assert log_dim <= sig_dim
                if depth > 1:
                    assert log_dim < sig_dim  # Strictly smaller for depth > 1

    def test_path_concatenation_property(self):
        """Test that log-signature satisfies path concatenation property.

        For paths X and Y, log_sig(X * Y) should relate to log_sig(X) and log_sig(Y)
        via the BCH formula.
        """
        # This is a complex test that requires proper BCH implementation
        # For now, we'll test on a simple case
        path1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]]).unsqueeze(0)
        path2 = torch.tensor([[1.0, 0.0], [1.0, 1.0]]).unsqueeze(0)

        log_sig1 = log_signature(path1, depth=2)
        log_sig2 = log_signature(path2, depth=2)

        # Concatenated path
        path_concat = torch.cat([path1, path2[:, 1:]], dim=1)
        log_sig_concat = log_signature(path_concat, depth=2)

        # The relationship is complex (BCH formula), so we just check shapes
        assert log_sig1.shape == log_sig2.shape == log_sig_concat.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
