# src/log_signatures_pytorch/triton_tensor_ops.py
from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# -------------------------
# Utilities
# -------------------------


def _prod(shape: Sequence[int]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return int(p)


# -------------------------
# Autotune configs
#   - BLOCK_S: timesteps per program (main lever for huge T)
#   - BLOCK_M: x-dim tile
#   - BLOCK_N: y-dim tile (tiny for width 3-5)
# -------------------------


def _outer_configs():
    # These are tuned for your regime:
    #   - width small -> N small -> keep BLOCK_N modest (8/16)
    #   - very long T -> increase BLOCK_S when M is small
    return [
        triton.Config({"BLOCK_S": 256, "BLOCK_M": 8, "BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_M": 16, "BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_M": 32, "BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_S": 32, "BLOCK_M": 64, "BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_S": 32, "BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4),
        # For larger M (e.g. width^3/width^4), reduce BLOCK_S to control registers
        triton.Config({"BLOCK_S": 16, "BLOCK_M": 64, "BLOCK_N": 8}, num_warps=4),
        triton.Config({"BLOCK_S": 16, "BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4),
    ]


# -------------------------
# Kernel: out = x ⊗ y
#
# Shapes passed in are (B, S, M) and (B, S, N).
# Output is (B, S, M, N).
#
# Grid:
#   pid0 = batch b
#   pid1 = s-block index
#   pid2 = flattened (m_tile, n_tile)
# -------------------------


@triton.autotune(configs=_outer_configs(), key=["M", "N"])
@triton.jit
def _bseq_outer_fwd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    stride_x_b: tl.constexpr,
    stride_x_s: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_y_b: tl.constexpr,
    stride_y_s: tl.constexpr,
    stride_y_n: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_m: tl.constexpr,
    stride_o_n: tl.constexpr,
    S: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sb = tl.program_id(1)
    pid_mn = tl.program_id(2)

    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // n_tiles
    pid_n = pid_mn - pid_m * n_tiles

    offs_s = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_s = offs_s < S
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Hint to compiler for better vectorization
    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_n, 8)

    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + offs_s[:, None] * stride_x_s
        + offs_m[None, :] * stride_x_m
    )

    x = tl.load(x_ptrs, mask=mask_s[:, None] & mask_m[None, :], other=0.0)

    # Store one n-lane at a time to avoid a (BLOCK_S,BLOCK_M,BLOCK_N) temp
    # NOTE: BLOCK_N is small (8/16), so this unroll is cheap.
    for nn in tl.static_range(0, BLOCK_N):
        n_idx = pid_n * BLOCK_N + nn  # Compute directly instead of indexing offs_n
        # Load y values for this n_idx directly (avoid constexpr indexing)
        y_n_ptrs = y_ptr + pid_b * stride_y_b + offs_s * stride_y_s + n_idx * stride_y_n
        y_nn = tl.load(y_n_ptrs, mask=mask_s & (n_idx < N), other=0.0)  # (BLOCK_S,)
        prod = x * y_nn[:, None]  # (BLOCK_S,BLOCK_M)

        o_ptrs = (
            out_ptr
            + pid_b * stride_o_b
            + offs_s[:, None] * stride_o_s
            + offs_m[None, :] * stride_o_m
            + n_idx * stride_o_n
        )
        tl.store(o_ptrs, prod, mask=mask_s[:, None] & mask_m[None, :] & (n_idx < N))


# -------------------------
# Kernel: out = base + x ⊗ y  (optionally scale y by y_scale)
#
# base/out are (B, S, M, N), x is (B, S, M), y is (B, S, N)
# -------------------------


@triton.autotune(configs=_outer_configs(), key=["M", "N"])
@triton.jit
def _bseq_add_outer_fwd_kernel(
    base_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    stride_b_b: tl.constexpr,
    stride_b_s: tl.constexpr,
    stride_b_m: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_s: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_y_b: tl.constexpr,
    stride_y_s: tl.constexpr,
    stride_y_n: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_m: tl.constexpr,
    stride_o_n: tl.constexpr,
    y_scale: tl.constexpr,  # compile-time scale is fastest; wrapper can specialize per divisor
    S: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sb = tl.program_id(1)
    pid_mn = tl.program_id(2)

    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // n_tiles
    pid_n = pid_mn - pid_m * n_tiles

    offs_s = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_s = offs_s < S
    mask_m = offs_m < M
    mask_n = offs_n < N

    tl.multiple_of(offs_m, 8)
    tl.multiple_of(offs_n, 8)

    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + offs_s[:, None] * stride_x_s
        + offs_m[None, :] * stride_x_m
    )

    x = tl.load(x_ptrs, mask=mask_s[:, None] & mask_m[None, :], other=0.0)

    for nn in tl.static_range(0, BLOCK_N):
        n_idx = pid_n * BLOCK_N + nn  # Compute directly instead of indexing offs_n
        # Load y values for this n_idx directly (avoid constexpr indexing)
        y_n_ptrs = y_ptr + pid_b * stride_y_b + offs_s * stride_y_s + n_idx * stride_y_n
        y_nn = (
            tl.load(y_n_ptrs, mask=mask_s & (n_idx < N), other=0.0) * y_scale
        )  # (BLOCK_S,)
        prod = x * y_nn[:, None]  # (BLOCK_S,BLOCK_M)

        b_ptrs = (
            base_ptr
            + pid_b * stride_b_b
            + offs_s[:, None] * stride_b_s
            + offs_m[None, :] * stride_b_m
            + n_idx * stride_b_n
        )
        base = tl.load(
            b_ptrs, mask=mask_s[:, None] & mask_m[None, :] & (n_idx < N), other=0.0
        )

        out = base + prod

        o_ptrs = (
            out_ptr
            + pid_b * stride_o_b
            + offs_s[:, None] * stride_o_s
            + offs_m[None, :] * stride_o_m
            + n_idx * stride_o_n
        )
        tl.store(o_ptrs, out, mask=mask_s[:, None] & mask_m[None, :] & (n_idx < N))


# -------------------------
# Python wrappers + autograd
# -------------------------


def _check_triton_ready(*tensors: Tensor):
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available/importable in this environment.")
    for t in tensors:
        if not t.is_cuda:
            raise RuntimeError("Triton kernels require CUDA tensors.")
    # float16/bf16/float32 are the usual safe set
    for t in tensors:
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"Unsupported dtype for Triton path: {t.dtype}")


def _flatten_bsm(t: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
    B, S = t.shape[:2]
    tail = tuple(t.shape[2:])
    M = _prod(tail) if len(tail) else 1
    return t.reshape(B, S, M), tail


class _BSeqOuterFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        _check_triton_ready(x, y)
        if x.shape[:2] != y.shape[:2]:
            raise ValueError(
                f"Expected shared (B,S), got {x.shape[:2]} vs {y.shape[:2]}"
            )

        B, S = x.shape[:2]
        x_flat, x_tail = _flatten_bsm(x)
        y_flat, y_tail = _flatten_bsm(y)
        M, N = x_flat.shape[2], y_flat.shape[2]

        # Enforce last dim contiguous (critical for performance / correctness assumptions)
        if x_flat.stride(2) != 1:
            x_flat = x_flat.contiguous()
        if y_flat.stride(2) != 1:
            y_flat = y_flat.contiguous()

        out = torch.empty((B, S, M, N), device=x.device, dtype=torch.result_type(x, y))

        def grid(meta):
            s_blocks = triton.cdiv(S, meta["BLOCK_S"])
            m_tiles = triton.cdiv(M, meta["BLOCK_M"])
            n_tiles = triton.cdiv(N, meta["BLOCK_N"])
            return (B, s_blocks, m_tiles * n_tiles)

        _bseq_outer_fwd_kernel[grid](
            x_flat,
            y_flat,
            out,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            y_flat.stride(0),
            y_flat.stride(1),
            y_flat.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            S=S,
            M=M,
            N=N,
        )

        ctx.save_for_backward(x_flat, y_flat)
        ctx.x_tail = x_tail
        ctx.y_tail = y_tail
        return out.reshape((B, S) + x_tail + y_tail)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_flat, y_flat = ctx.saved_tensors
        B, S, M = x_flat.shape
        N = y_flat.shape[2]

        g = grad_out.reshape(B, S, M, N)

        dx = dy = None
        if ctx.needs_input_grad[0]:
            dx_flat = torch.matmul(g, y_flat.unsqueeze(-1)).squeeze(-1)  # (B,S,M)
            dx = dx_flat.reshape((B, S) + ctx.x_tail)
        if ctx.needs_input_grad[1]:
            dy_flat = torch.matmul(g.transpose(-2, -1), x_flat.unsqueeze(-1)).squeeze(
                -1
            )  # (B,S,N)
            dy = dy_flat.reshape((B, S) + ctx.y_tail)
        return dx, dy


class _BSeqAddOuterFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, base: Tensor, x: Tensor, y: Tensor, y_scale: float = 1.0
    ) -> Tensor:
        _check_triton_ready(base, x, y)
        if base.shape[:2] != x.shape[:2] or base.shape[:2] != y.shape[:2]:
            raise ValueError("base, x, y must share (B,S)")

        B, S = x.shape[:2]
        x_flat, x_tail = _flatten_bsm(x)
        y_flat, y_tail = _flatten_bsm(y)
        M, N = x_flat.shape[2], y_flat.shape[2]

        # base must be shape (B,S,*x_tail,*y_tail)
        expected_tail = x_tail + y_tail
        if tuple(base.shape[2:]) != expected_tail:
            raise ValueError(
                f"base trailing shape must be {expected_tail}, got {tuple(base.shape[2:])}"
            )

        base_flat = base.reshape(B, S, M, N)

        if x_flat.stride(2) != 1:
            x_flat = x_flat.contiguous()
        if y_flat.stride(2) != 1:
            y_flat = y_flat.contiguous()

        # Promote dtypes: result_type handles promotion, but type checker needs help
        dtype = base.dtype
        if x.dtype != dtype or y.dtype != dtype:
            # In practice, all should match, but handle promotion if needed
            dtype = torch.promote_types(
                torch.promote_types(base.dtype, x.dtype), y.dtype
            )
        out = torch.empty((B, S, M, N), device=base.device, dtype=dtype)

        # Specialize y_scale as constexpr by passing it as a meta-parameter (fastest).
        # This will compile a few variants; in your use case y_scale is one of {1/2,1/3,1/4}.
        y_scale_meta = float(y_scale)

        def grid(meta):
            s_blocks = triton.cdiv(S, meta["BLOCK_S"])
            m_tiles = triton.cdiv(M, meta["BLOCK_M"])
            n_tiles = triton.cdiv(N, meta["BLOCK_N"])
            return (B, s_blocks, m_tiles * n_tiles)

        _bseq_add_outer_fwd_kernel[grid](
            base_flat,
            x_flat,
            y_flat,
            out,
            base_flat.stride(0),
            base_flat.stride(1),
            base_flat.stride(2),
            base_flat.stride(3),
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            y_flat.stride(0),
            y_flat.stride(1),
            y_flat.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            y_scale=y_scale_meta,
            S=S,
            M=M,
            N=N,
        )

        ctx.save_for_backward(x_flat, y_flat)
        ctx.base_shape = base.shape
        ctx.x_tail = x_tail
        ctx.y_tail = y_tail
        return out.reshape(ctx.base_shape)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x_flat, y_flat = ctx.saved_tensors
        B, S, M = x_flat.shape
        N = y_flat.shape[2]

        g = grad_out.reshape(B, S, M, N)

        dbase = dx = dy = None
        if ctx.needs_input_grad[0]:
            dbase = grad_out
        if ctx.needs_input_grad[1]:
            dx_flat = torch.matmul(g, y_flat.unsqueeze(-1)).squeeze(-1)  # (B,S,M)
            dx = dx_flat.reshape((B, S) + ctx.x_tail)
        if ctx.needs_input_grad[2]:
            dy_flat = torch.matmul(g.transpose(-2, -1), x_flat.unsqueeze(-1)).squeeze(
                -1
            )  # (B,S,N)
            dy = dy_flat.reshape((B, S) + ctx.y_tail)
        # y_scale is a python float, no grad
        return dbase, dx, dy, None


def batch_sequence_tensor_product_triton(x: Tensor, y: Tensor) -> Tensor:
    """(B,S,...) ⊗ (B,S,...) -> (B,S,...,...), Triton-optimized."""
    return _BSeqOuterFn.apply(x, y)


def batch_sequence_add_tensor_product_triton(
    base: Tensor, x: Tensor, y: Tensor, *, y_scale: float = 1.0
) -> Tensor:
    """base + x⊗y, preserving (B,S), Triton-optimized."""
    return _BSeqAddOuterFn.apply(base, x, y, float(y_scale))


# -------------------------
# Main function for testing/example usage
# -------------------------


def main() -> None:
    """Run example computations to test Triton kernels in isolation."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping Triton kernel examples.")
        return

    if not _TRITON_AVAILABLE:
        print("Triton not available. Skipping Triton kernel examples.")
        return

    print("=" * 60)
    print("Triton Kernel Example Computations")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.float32

    # Test 1: batch_sequence_tensor_product_triton
    print("\n[Test 1] batch_sequence_tensor_product_triton")
    print("-" * 60)
    B, S, M, N = 2, 5, 3, 4
    x = torch.randn(B, S, M, device=device, dtype=dtype)
    y = torch.randn(B, S, N, device=device, dtype=dtype)

    result_triton = batch_sequence_tensor_product_triton(x, y)
    print(f"Input shapes: x={x.shape}, y={y.shape}")
    print(f"Output shape: {result_triton.shape}")
    print(f"Expected shape: ({B}, {S}, {M}, {N})")

    # Compare with reference implementation (CPU)
    from log_signatures_pytorch.tensor_ops import batch_sequence_tensor_product

    x_cpu = x.cpu()
    y_cpu = y.cpu()
    result_ref = batch_sequence_tensor_product(x_cpu, y_cpu)
    result_triton_cpu = result_triton.cpu()

    max_diff = (result_triton_cpu - result_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("✓ Results match reference implementation")
    else:
        print("⚠ Warning: difference exceeds tolerance")

    # Test 2: batch_sequence_add_tensor_product_triton
    print("\n[Test 2] batch_sequence_add_tensor_product_triton")
    print("-" * 60)
    base = torch.randn(B, S, M, N, device=device, dtype=dtype)
    x2 = torch.randn(B, S, M, device=device, dtype=dtype)
    y2 = torch.randn(B, S, N, device=device, dtype=dtype)
    y_scale = 0.5

    result_triton2 = batch_sequence_add_tensor_product_triton(
        base, x2, y2, y_scale=y_scale
    )
    print(f"Input shapes: base={base.shape}, x={x2.shape}, y={y2.shape}")
    print(f"Output shape: {result_triton2.shape}")
    print(f"y_scale: {y_scale}")

    # Manual reference: base + (x ⊗ (y * y_scale))
    base_cpu = base.cpu()
    x2_cpu = x2.cpu()
    y2_cpu = y2.cpu()
    manual_ref = base_cpu + batch_sequence_tensor_product(x2_cpu, y2_cpu * y_scale)
    result_triton2_cpu = result_triton2.cpu()

    max_diff2 = (result_triton2_cpu - manual_ref).abs().max().item()
    print(f"Max difference vs manual reference: {max_diff2:.2e}")
    if max_diff2 < 1e-5:
        print("✓ Results match manual reference")
    else:
        print("⚠ Warning: difference exceeds tolerance")

    # Test 3: Gradient computation
    print("\n[Test 3] Gradient computation")
    print("-" * 60)
    x_grad = torch.randn(B, S, M, device=device, dtype=dtype, requires_grad=True)
    y_grad = torch.randn(B, S, N, device=device, dtype=dtype, requires_grad=True)

    out_grad = batch_sequence_tensor_product_triton(x_grad, y_grad)
    loss = out_grad.sum()
    loss.backward()

    print(f"x.grad shape: {x_grad.grad.shape if x_grad.grad is not None else None}")
    print(f"y.grad shape: {y_grad.grad.shape if y_grad.grad is not None else None}")
    if x_grad.grad is not None and y_grad.grad is not None:
        print(f"x.grad finite: {torch.isfinite(x_grad.grad).all().item()}")
        print(f"y.grad finite: {torch.isfinite(y_grad.grad).all().item()}")
        print("✓ Gradients computed successfully")

    # Test 4: Different shapes and edge cases
    print("\n[Test 4] Edge cases")
    print("-" * 60)

    # Small shapes
    x_small = torch.randn(1, 1, 2, device=device, dtype=dtype)
    y_small = torch.randn(1, 1, 3, device=device, dtype=dtype)
    result_small = batch_sequence_tensor_product_triton(x_small, y_small)
    print(f"Small shapes (1,1,2) ⊗ (1,1,3): {result_small.shape}")

    # Larger batch/sequence
    x_large = torch.randn(4, 10, 5, device=device, dtype=dtype)
    y_large = torch.randn(4, 10, 6, device=device, dtype=dtype)
    result_large = batch_sequence_tensor_product_triton(x_large, y_large)
    print(f"Larger shapes (4,10,5) ⊗ (4,10,6): {result_large.shape}")

    # Different y_scale values
    base_scale = torch.randn(2, 3, 4, 5, device=device, dtype=dtype)
    x_scale = torch.randn(2, 3, 4, device=device, dtype=dtype)
    y_scale_tensor = torch.randn(2, 3, 5, device=device, dtype=dtype)
    for scale_val in [0.25, 0.5, 1.0, 2.0]:
        result_scale = batch_sequence_add_tensor_product_triton(
            base_scale, x_scale, y_scale_tensor, y_scale=scale_val
        )
        print(f"y_scale={scale_val}: output shape {result_scale.shape}")

    print("\n" + "=" * 60)
    print("All example computations completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
