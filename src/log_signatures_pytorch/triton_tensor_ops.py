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
# -------------------------


def _outer_configs():
    # Optimized for coalesced N-dimension access.
    # We include BLOCK_N=16 for small width cases (N=9, 27).
    # We include num_warps=8 to hide latency.
    return [
        # Small N (e.g. 9), Large S
        triton.Config({"BLOCK_S": 128, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=8),
        
        # Medium N (e.g. 27), Large S
        triton.Config({"BLOCK_S": 128, "BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        
        # Larger sizes
        triton.Config({"BLOCK_S": 64, "BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 32, "BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        
        # Fallback
        triton.Config({"BLOCK_S": 32, "BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4),
    ]


# -------------------------
# Kernels: 2D Loop-Swapped (Optimized)
# -------------------------


@triton.autotune(configs=_outer_configs(), key=["M", "N"])
@triton.jit
def _bseq_outer_fwd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    stride_x_b: tl.constexpr, stride_x_s: tl.constexpr, stride_x_m: tl.constexpr,
    stride_y_b: tl.constexpr, stride_y_s: tl.constexpr, stride_y_n: tl.constexpr,
    stride_o_b: tl.constexpr, stride_o_s: tl.constexpr, stride_o_m: tl.constexpr, stride_o_n: tl.constexpr,
    S: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sb = tl.program_id(1)
    pid_mn = tl.program_id(2)

    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // n_tiles
    pid_n = pid_mn - pid_m * n_tiles

    offs_s = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_s = offs_s < S
    mask_n = offs_n < N

    # Hint to compiler for better vectorization along contiguous N
    tl.multiple_of(offs_n, 8)

    # Load Y block (BLOCK_S, BLOCK_N) once - reused for all M columns
    y_ptrs = (
        y_ptr
        + pid_b * stride_y_b
        + offs_s[:, None] * stride_y_s
        + offs_n[None, :] * stride_y_n
    )
    y = tl.load(y_ptrs, mask=mask_s[:, None] & mask_n[None, :], other=0.0)

    # Loop over columns of X (BLOCK_M) to allow contiguous N-writes
    for mm in tl.static_range(0, BLOCK_M):
        m_idx = pid_m * BLOCK_M + mm
        
        # Load X column (BLOCK_S,)
        x_ptrs = (
            x_ptr 
            + pid_b * stride_x_b 
            + offs_s * stride_x_s 
            + m_idx * stride_x_m
        )
        # Check m_idx < M in the mask
        x_col = tl.load(x_ptrs, mask=mask_s & (m_idx < M), other=0.0)

        # Compute outer product slice: (BLOCK_S, 1) * (BLOCK_S, BLOCK_N)
        prod = x_col[:, None] * y

        # Store result (BLOCK_S, BLOCK_N) - Contiguous along N
        o_ptrs = (
            out_ptr
            + pid_b * stride_o_b
            + offs_s[:, None] * stride_o_s
            + m_idx * stride_o_m
            + offs_n[None, :] * stride_o_n
        )
        tl.store(o_ptrs, prod, mask=mask_s[:, None] & mask_n[None, :] & (m_idx < M))


@triton.autotune(configs=_outer_configs(), key=["M", "N"])
@triton.jit
def _bseq_add_outer_fwd_kernel(
    base_ptr,
    x_ptr,
    y_ptr,
    out_ptr,
    stride_b_b: tl.constexpr, stride_b_s: tl.constexpr, stride_b_m: tl.constexpr, stride_b_n: tl.constexpr,
    stride_x_b: tl.constexpr, stride_x_s: tl.constexpr, stride_x_m: tl.constexpr,
    stride_y_b: tl.constexpr, stride_y_s: tl.constexpr, stride_y_n: tl.constexpr,
    stride_o_b: tl.constexpr, stride_o_s: tl.constexpr, stride_o_m: tl.constexpr, stride_o_n: tl.constexpr,
    y_scale: tl.constexpr,
    S: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sb = tl.program_id(1)
    pid_mn = tl.program_id(2)

    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // n_tiles
    pid_n = pid_mn - pid_m * n_tiles

    offs_s = pid_sb * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_s = offs_s < S
    mask_n = offs_n < N

    tl.multiple_of(offs_n, 8)

    # Load Y block once (reused)
    y_ptrs = (
        y_ptr
        + pid_b * stride_y_b
        + offs_s[:, None] * stride_y_s
        + offs_n[None, :] * stride_y_n
    )
    y = tl.load(y_ptrs, mask=mask_s[:, None] & mask_n[None, :], other=0.0) * y_scale

    # Loop over columns of X
    for mm in tl.static_range(0, BLOCK_M):
        m_idx = pid_m * BLOCK_M + mm

        # Load X column
        x_ptrs = (
            x_ptr
            + pid_b * stride_x_b
            + offs_s * stride_x_s
            + m_idx * stride_x_m
        )
        x_col = tl.load(x_ptrs, mask=mask_s & (m_idx < M), other=0.0)

        # Compute product
        prod = x_col[:, None] * y

        # Load Base slice (Contiguous N)
        b_ptrs = (
            base_ptr
            + pid_b * stride_b_b
            + offs_s[:, None] * stride_b_s
            + m_idx * stride_b_m
            + offs_n[None, :] * stride_b_n
        )
        base = tl.load(b_ptrs, mask=mask_s[:, None] & mask_n[None, :] & (m_idx < M), other=0.0)

        out = base + prod

        # Store Out (Contiguous N)
        o_ptrs = (
            out_ptr
            + pid_b * stride_o_b
            + offs_s[:, None] * stride_o_s
            + m_idx * stride_o_m
            + offs_n[None, :] * stride_o_n
        )
        tl.store(o_ptrs, out, mask=mask_s[:, None] & mask_n[None, :] & (m_idx < M))


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
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            y_flat.stride(0), y_flat.stride(1), y_flat.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            S=S, M=M, N=N,
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
            dx_flat = torch.matmul(g, y_flat.unsqueeze(-1)).squeeze(-1)
            dx = dx_flat.reshape((B, S) + ctx.x_tail)
        if ctx.needs_input_grad[1]:
            dy_flat = torch.matmul(g.transpose(-2, -1), x_flat.unsqueeze(-1)).squeeze(-1)
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

        # Promote dtypes
        dtype = base.dtype
        if x.dtype != dtype or y.dtype != dtype:
            dtype = torch.promote_types(
                torch.promote_types(base.dtype, x.dtype), y.dtype
            )
        out = torch.empty((B, S, M, N), device=base.device, dtype=dtype)

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
            base_flat.stride(0), base_flat.stride(1), base_flat.stride(2), base_flat.stride(3),
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            y_flat.stride(0), y_flat.stride(1), y_flat.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            y_scale=y_scale_meta,
            S=S, M=M, N=N,
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
            dx_flat = torch.matmul(g, y_flat.unsqueeze(-1)).squeeze(-1)
            dx = dx_flat.reshape((B, S) + ctx.x_tail)
        if ctx.needs_input_grad[2]:
            dy_flat = torch.matmul(g.transpose(-2, -1), x_flat.unsqueeze(-1)).squeeze(
                -1
            )
            dy = dy_flat.reshape((B, S) + ctx.y_tail)
        return dbase, dx, dy, None


def batch_sequence_tensor_product_triton(x: Tensor, y: Tensor) -> Tensor:
    """(B,S,...) ⊗ (B,S,...) -> (B,S,...,...), Triton-optimized."""
    return _BSeqOuterFn.apply(x, y)


def batch_sequence_add_tensor_product_triton(
    base: Tensor, x: Tensor, y: Tensor, *, y_scale: float = 1.0
) -> Tensor:
    """base + x⊗y, preserving (B,S), Triton-optimized."""
    return _BSeqAddOuterFn.apply(base, x, y, float(y_scale))


def main() -> None:
    """Run example computations to test Triton kernels in isolation."""
    if not torch.cuda.is_available():
        return
    if not _TRITON_AVAILABLE:
        return

    print("Running Triton Optimized 2D Kernel Test")
    device = torch.device("cuda")
    B, S, M, N = 2, 5, 3, 4
    x = torch.randn(B, S, M, device=device)
    y = torch.randn(B, S, N, device=device)
    
    # Test product
    res = batch_sequence_tensor_product_triton(x, y)
    print(f"Product shape: {res.shape}")
    
    # Test add
    base = torch.randn(B, S, M, N, device=device)
    res_add = batch_sequence_add_tensor_product_triton(base, x, y)
    print(f"Add shape: {res_add.shape}")

if __name__ == "__main__":
    main()
