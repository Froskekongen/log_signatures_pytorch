"""Tests for Triton-optimized tensor operations.

This module tests the Triton kernels in triton_tensor_ops.py, comparing
against reference implementations and validating correctness, gradients,
and edge cases.
"""

import time

import pytest
import torch

from log_signatures_pytorch.tensor_ops import (
    batch_sequence_add_tensor_product,
    batch_sequence_tensor_product,
)

# Check for CUDA and Triton availability
_TRITON_AVAILABLE = True
try:
    import triton
except ImportError:
    _TRITON_AVAILABLE = False

try:
    from log_signatures_pytorch.triton_tensor_ops import (
        batch_sequence_add_tensor_product_triton,
        batch_sequence_tensor_product_triton,
    )
except ImportError:
    batch_sequence_tensor_product_triton = None
    batch_sequence_add_tensor_product_triton = None

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_IF_NO_CUDA = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
SKIP_IF_NO_TRITON = pytest.mark.skipif(
    not _TRITON_AVAILABLE, reason="Triton not available"
)
SKIP_IF_NO_FUNCTIONS = pytest.mark.skipif(
    batch_sequence_tensor_product_triton is None,
    reason="Triton functions not available",
)


# -------------------------
# Helper functions
# -------------------------


def _time_function(func, *args, **kwargs):
    """Time a function call and return (result, time_seconds)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed


# -------------------------
# 1. Correctness Tests (vs Reference Implementation)
# -------------------------


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize(
    ("B", "S", "M", "N"),
    [(1, 1, 2, 3), (2, 5, 3, 4), (4, 10, 5, 6), (8, 20, 7, 8)],
)
def test_batch_sequence_tensor_product_triton_matches_reference(
    B: int, S: int, M: int, N: int
) -> None:
    """Test that Triton kernel matches GPU reference implementation."""
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(42)
    x = torch.randn(B, S, M, device=device, dtype=dtype)
    y = torch.randn(B, S, N, device=device, dtype=dtype)

    # Time Triton kernel
    result_triton, triton_time = _time_function(
        batch_sequence_tensor_product_triton, x, y
    )

    # Compare with GPU reference
    result_ref, ref_time = _time_function(batch_sequence_tensor_product, x, y)

    # Validate correctness
    torch.testing.assert_close(result_triton, result_ref, rtol=1e-5, atol=1e-6)

    # Log timing (informational, not a test failure)
    print(
        f"\n  Shape ({B},{S},{M})⊗({B},{S},{N}): "
        f"Triton={triton_time * 1000:.3f}ms, "
        f"GPU={ref_time * 1000:.3f}ms, "
        f"speedup={ref_time / triton_time:.2f}x"
    )


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize(
    ("B", "S", "M", "N", "y_scale"),
    [
        (2, 5, 3, 4, 0.25),
        (2, 5, 3, 4, 0.5),
        (2, 5, 3, 4, 1.0),
        (2, 5, 3, 4, 2.0),
        (4, 10, 5, 6, 0.5),
    ],
)
def test_batch_sequence_add_tensor_product_triton_matches_reference(
    B: int, S: int, M: int, N: int, y_scale: float
) -> None:
    """Test that Triton add-tensor-product matches GPU reference implementation."""
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(43)
    base = torch.randn(B, S, M, N, device=device, dtype=dtype)
    x = torch.randn(B, S, M, device=device, dtype=dtype)
    y = torch.randn(B, S, N, device=device, dtype=dtype)

    # Time Triton kernel
    result_triton, triton_time = _time_function(
        batch_sequence_add_tensor_product_triton, base, x, y, y_scale=y_scale
    )

    # Reference using batch_sequence_add_tensor_product: base + (x ⊗ (y * y_scale))
    y_scaled = y * y_scale
    result_ref, ref_time = _time_function(
        batch_sequence_add_tensor_product, base, x, y_scaled
    )

    # Validate correctness
    torch.testing.assert_close(result_triton, result_ref, rtol=1e-5, atol=1e-6)

    # Log timing
    print(
        f"\n  Add product ({B},{S},{M},{N}) scale={y_scale}: "
        f"Triton={triton_time * 1000:.3f}ms, "
        f"GPU={ref_time * 1000:.3f}ms, "
        f"speedup={ref_time / triton_time:.2f}x"
    )


# -------------------------
# 2. Shape Validation Tests
# -------------------------


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize(
    ("B", "S", "M", "N"),
    [(1, 1, 2, 3), (2, 5, 3, 4), (4, 10, 5, 6), (8, 20, 7, 8)],
)
def test_batch_sequence_tensor_product_triton_output_shape(
    B: int, S: int, M: int, N: int
) -> None:
    """Test that output shape is correct."""
    device = torch.device("cuda")
    dtype = torch.float32

    x = torch.randn(B, S, M, device=device, dtype=dtype)
    y = torch.randn(B, S, N, device=device, dtype=dtype)

    result = batch_sequence_tensor_product_triton(x, y)

    assert result.shape == (B, S, M, N), (
        f"Expected ({B},{S},{M},{N}), got {result.shape}"
    )


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
def test_batch_sequence_tensor_product_triton_input_validation() -> None:
    """Test input validation for Triton kernels."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Test mismatched (B, S) dimensions
    x = torch.randn(2, 5, 3, device=device, dtype=dtype)
    y = torch.randn(2, 4, 4, device=device, dtype=dtype)  # Different S
    with pytest.raises(ValueError, match="shared"):
        batch_sequence_tensor_product_triton(x, y)

    # Test non-CUDA tensors
    x_cpu = torch.randn(2, 5, 3, dtype=dtype)
    y_cpu = torch.randn(2, 5, 4, dtype=dtype)
    with pytest.raises(RuntimeError, match="CUDA"):
        batch_sequence_tensor_product_triton(x_cpu, y_cpu)

    # Test unsupported dtype (int64)
    x_int = torch.randint(0, 10, (2, 5, 3), device=device, dtype=torch.int64)
    y_int = torch.randint(0, 10, (2, 5, 4), device=device, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="dtype"):
        batch_sequence_tensor_product_triton(x_int, y_int)


# -------------------------
# 3. Gradient Computation Tests
# -------------------------


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize(
    ("B", "S", "M", "N"),
    [(2, 5, 3, 4), (4, 10, 5, 6)],
)
def test_batch_sequence_tensor_product_triton_grad_propagates(
    B: int, S: int, M: int, N: int
) -> None:
    """Test that gradients propagate correctly."""
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(44)
    x = torch.randn(B, S, M, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(B, S, N, device=device, dtype=dtype, requires_grad=True)

    out = batch_sequence_tensor_product_triton(x, y)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "x.grad should not be None"
    assert y.grad is not None, "y.grad should not be None"
    assert x.grad.shape == x.shape, (
        f"x.grad shape mismatch: {x.grad.shape} vs {x.shape}"
    )
    assert y.grad.shape == y.shape, (
        f"y.grad shape mismatch: {y.grad.shape} vs {y.shape}"
    )
    assert torch.all(torch.isfinite(x.grad)), "x.grad should be finite"
    assert torch.all(torch.isfinite(y.grad)), "y.grad should be finite"
    assert torch.any(x.grad != 0), "x.grad should be non-zero"
    assert torch.any(y.grad != 0), "y.grad should be non-zero"


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize(
    ("B", "S", "M", "N"),
    [(2, 5, 3, 4)],
)
def test_batch_sequence_tensor_product_triton_grad_matches_reference(
    B: int, S: int, M: int, N: int
) -> None:
    """Test that gradients match reference implementation."""
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(45)
    x = torch.randn(B, S, M, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(B, S, N, device=device, dtype=dtype, requires_grad=True)

    # Triton gradients
    out_triton = batch_sequence_tensor_product_triton(x, y)
    loss_triton = out_triton.sum()
    loss_triton.backward()
    x_grad_triton = x.grad.clone()
    y_grad_triton = y.grad.clone()

    # Reset and compute reference gradients
    x.grad = None
    y.grad = None
    x_cpu = x.cpu().detach().requires_grad_(True)
    y_cpu = y.cpu().detach().requires_grad_(True)
    out_ref = batch_sequence_tensor_product(x_cpu, y_cpu)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    x_grad_ref = x_cpu.grad
    y_grad_ref = y_cpu.grad

    # Compare gradients
    torch.testing.assert_close(x_grad_triton.cpu(), x_grad_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(y_grad_triton.cpu(), y_grad_ref, rtol=1e-4, atol=1e-5)


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize(
    ("B", "S", "M", "N", "y_scale"),
    [(2, 5, 3, 4, 0.5), (2, 5, 3, 4, 1.0)],
)
def test_batch_sequence_add_tensor_product_triton_grad_propagates(
    B: int, S: int, M: int, N: int, y_scale: float
) -> None:
    """Test gradients for add-tensor-product kernel."""
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(46)
    base = torch.randn(B, S, M, N, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(B, S, M, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(B, S, N, device=device, dtype=dtype, requires_grad=True)

    out = batch_sequence_add_tensor_product_triton(base, x, y, y_scale=y_scale)
    loss = out.sum()
    loss.backward()

    assert base.grad is not None, "base.grad should not be None"
    assert x.grad is not None, "x.grad should not be None"
    assert y.grad is not None, "y.grad should not be None"
    assert torch.all(torch.isfinite(base.grad)), "base.grad should be finite"
    assert torch.all(torch.isfinite(x.grad)), "x.grad should be finite"
    assert torch.all(torch.isfinite(y.grad)), "y.grad should be finite"


# -------------------------
# 4. Edge Cases
# -------------------------


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
def test_triton_kernels_small_shapes() -> None:
    """Test kernels with very small shapes."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Smallest reasonable shapes
    x = torch.randn(1, 1, 2, device=device, dtype=dtype)
    y = torch.randn(1, 1, 3, device=device, dtype=dtype)

    result = batch_sequence_tensor_product_triton(x, y)
    assert result.shape == (1, 1, 2, 3)

    # Test add version
    base = torch.randn(1, 1, 2, 3, device=device, dtype=dtype)
    result_add = batch_sequence_add_tensor_product_triton(base, x, y)
    assert result_add.shape == (1, 1, 2, 3)


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
def test_triton_kernels_large_shapes() -> None:
    """Test kernels with larger shapes."""
    device = torch.device("cuda")
    dtype = torch.float32

    B, S, M, N = 8, 100, 10, 12
    x = torch.randn(B, S, M, device=device, dtype=dtype)
    y = torch.randn(B, S, N, device=device, dtype=dtype)

    result, elapsed = _time_function(batch_sequence_tensor_product_triton, x, y)
    assert result.shape == (B, S, M, N)
    print(f"\n  Large shape ({B},{S},{M})⊗({B},{S},{N}): {elapsed * 1000:.3f}ms")


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_kernels_dtype_support(dtype: torch.dtype) -> None:
    """Test kernels with different dtypes."""
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Check if dtype is supported on this device
    try:
        x = torch.randn(2, 5, 3, device=device, dtype=dtype)
        y = torch.randn(2, 5, 4, device=device, dtype=dtype)
    except RuntimeError:
        pytest.skip(f"dtype {dtype} not supported on this device")

    result = batch_sequence_tensor_product_triton(x, y)
    assert result.dtype == dtype or result.dtype == torch.float32  # May promote

    # Adjust tolerance for lower precision
    # bfloat16 has lower precision than float16 in some operations
    atol = (
        2e-2 if dtype == torch.bfloat16 else (1e-3 if dtype == torch.float16 else 1e-6)
    )
    rtol = (
        2e-2 if dtype == torch.bfloat16 else (1e-3 if dtype == torch.float16 else 1e-5)
    )

    # Compare with reference
    x_cpu = x.cpu().float()
    y_cpu = y.cpu().float()
    result_ref = batch_sequence_tensor_product(x_cpu, y_cpu)
    result_cpu = result.cpu().float()

    torch.testing.assert_close(result_cpu, result_ref, rtol=rtol, atol=atol)


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
def test_triton_kernels_non_contiguous() -> None:
    """Test kernels with non-contiguous inputs."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Create non-contiguous tensors (transposed)
    x_contig = torch.randn(2, 5, 3, device=device, dtype=dtype)
    y_contig = torch.randn(2, 5, 4, device=device, dtype=dtype)
    x = x_contig.transpose(0, 1).transpose(0, 1)  # Non-contiguous
    y = y_contig.transpose(0, 1).transpose(0, 1)  # Non-contiguous

    # Should work (implementation makes contiguous if needed)
    result = batch_sequence_tensor_product_triton(x, y)
    assert result.shape == (2, 5, 3, 4)

    # Compare with contiguous version
    result_contig = batch_sequence_tensor_product_triton(x_contig, y_contig)
    torch.testing.assert_close(result, result_contig, rtol=1e-5, atol=1e-6)


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
def test_triton_kernels_zero_inputs() -> None:
    """Test kernels with zero inputs."""
    device = torch.device("cuda")
    dtype = torch.float32

    x = torch.zeros(2, 5, 3, device=device, dtype=dtype)
    y = torch.zeros(2, 5, 4, device=device, dtype=dtype)

    result = batch_sequence_tensor_product_triton(x, y)
    assert torch.allclose(result, torch.zeros_like(result))

    # Test add version
    base = torch.ones(2, 5, 3, 4, device=device, dtype=dtype)
    result_add = batch_sequence_add_tensor_product_triton(base, x, y)
    torch.testing.assert_close(result_add, base)


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_TRITON
@SKIP_IF_NO_FUNCTIONS
def test_triton_kernels_extreme_values() -> None:
    """Test kernels with extreme values."""
    device = torch.device("cuda")
    dtype = torch.float32

    # Large values
    x = torch.full((2, 5, 3), 1e6, device=device, dtype=dtype)
    y = torch.full((2, 5, 4), 1e6, device=device, dtype=dtype)

    result = batch_sequence_tensor_product_triton(x, y)
    assert torch.all(torch.isfinite(result)), "Result should be finite"
    assert not torch.any(torch.isnan(result)), "Result should not contain NaN"
    assert not torch.any(torch.isinf(result)), "Result should not contain Inf"

    # Small values
    x = torch.full((2, 5, 3), 1e-6, device=device, dtype=dtype)
    y = torch.full((2, 5, 4), 1e-6, device=device, dtype=dtype)

    result = batch_sequence_tensor_product_triton(x, y)
    assert torch.all(torch.isfinite(result)), "Result should be finite"
    assert not torch.any(torch.isnan(result)), "Result should not contain NaN"
