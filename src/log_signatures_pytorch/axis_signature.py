import torch
from torch import Tensor


def extract_sparse_events(
    inc: Tensor,
    gmax: int,
    eps: float = 0.0,
    selection: str = "topk",
    order: str = "sorted",
    check_exact: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """
    Extract sparse changes (indices and values) from path increments.

    Parameters
    ----------
    inc : Tensor
        Increments of shape (B, T, d).
    gmax : int
        Maximum number of changing coordinates per step.
    eps : float
        Threshold for considering a change as zero.
    selection : str
        Method to select changing coordinates: "topk".
        ("threshold" not fully implemented in this version as topk is preferred for fixed tensor shapes).
    order : str
        Order of coordinates within a step: "sorted", "reverse", "given" (by topk score).
    check_exact : bool
        If True, returns nnz count to verify if gmax was sufficient.

    Returns
    -------
    idx : Tensor
        Indices of changing coordinates, shape (B, T, gmax).
    delta : Tensor
        Values of changes, shape (B, T, gmax).
    nnz : Tensor | None
        Number of non-zeros per step (B, T) if check_exact is True, else None.
    """
    B, T, d = inc.shape

    # 1. Zero out small increments
    if eps > 0:
        inc = torch.where(inc.abs() > eps, inc, inc.new_zeros(()))

    nnz = None
    if check_exact:
        nnz = (inc.abs() > 0).sum(dim=-1)
        if (nnz > gmax).any():
            # We just calculate it here, the caller can decide to raise or warn.
            # Or we can just return it. The plan suggested raising, but returning allows caller control.
            pass

    # 2. Select indices
    if selection == "topk":
        # values, indices
        # We use topk on abs values to find most significant changes
        top_vals, idx = inc.abs().topk(k=gmax, dim=-1)
        # Gather actual values (signed)
        delta = torch.gather(inc, -1, idx)
    else:
        raise NotImplementedError(
            f"Selection '{selection}' not implemented yet. Use 'topk'."
        )

    # 3. Order coordinates
    if order == "sorted":
        idx, sort_indices = idx.sort(dim=-1)
        delta = torch.gather(delta, -1, sort_indices)
    elif order == "reverse":
        idx, sort_indices = idx.sort(dim=-1, descending=True)
        delta = torch.gather(delta, -1, sort_indices)
    elif order == "given":
        pass  # Keep topk order (largest magnitude first)
    else:
        raise ValueError(f"Unknown order '{order}'.")

    return idx, delta, nnz


def signature_axis(
    path: Tensor,
    depth: int,
    stream: bool = False,
    gmax: int | None = None,
    eps: float = 0.0,
    selection: str = "topk",
    order: str = "sorted",
    check_exact: bool = True,
) -> Tensor:
    """
    Compute signature using axis-path (rectilinear) interpolation.

    Parameters
    ----------
    path : Tensor
        Input path of shape (B, L, d).
    depth : int
        Signature depth (1, 2, or 3).
    stream : bool
        If True, return signature at every original step.
    gmax : int | None
        Max active coordinates per step. Defaults to ceil(0.01 * d) if None.
    eps : float
        Threshold for zero increments.
    selection : str
        "topk" or "threshold".
    order : str
        "sorted", "reverse", or "given".
    check_exact : bool
        If True, raises RuntimeError if more than gmax coordinates change in a step.

    Returns
    -------
    Tensor
        Signature tensor.
        Shape (B, sig_dim) if stream=False.
        Shape (B, L-1, sig_dim) if stream=True.
        sig_dim is sum(d^k for k=1..depth).
    """
    if path.ndim != 3:
        raise ValueError("path must be (B, L, d)")
    if depth not in (1, 2, 3):
        raise NotImplementedError(f"Depth {depth} not implemented for axis signature.")

    B, L, d = path.shape
    if L < 2:
        raise ValueError("Path length must be at least 2.")

    # Determine gmax
    if gmax is None:
        gmax = (
            max(1, int(np.ceil(0.01 * d)))
            if "np" in locals()
            else max(1, int(0.01 * d) + 1)
        )

    # Calculate increments
    inc = path[:, 1:] - path[:, :-1]  # (B, T, d) where T = L-1
    T = L - 1

    # Extract sparse events
    idx, delta, nnz = extract_sparse_events(
        inc, gmax, eps, selection, order, check_exact
    )

    if check_exact and nnz is not None:
        if (nnz > gmax).any():
            max_nnz = nnz.max().item()
            raise RuntimeError(
                f"Axis signature precision loss: found {max_nnz} active coordinates, "
                f"but gmax={gmax}. Increase gmax or verify sparsity."
            )

    # Initialize signature levels
    # We maintain flat tensors for efficiency

    # T1: (B, d)
    t1 = torch.zeros((B, d), device=path.device, dtype=path.dtype)

    levels = [t1]

    if depth >= 2:
        # T2: (B, d*d)
        t2 = torch.zeros((B, d * d), device=path.device, dtype=path.dtype)
        levels.append(t2)
        # Precompute indexing helpers
        row_idx_d = torch.arange(d, device=path.device, dtype=torch.long) * d

    if depth >= 3:
        # T3: (B, d*d*d)
        t3 = torch.zeros((B, d * d * d), device=path.device, dtype=path.dtype)
        levels.append(t3)
        # Precompute indexing helpers
        row_idx_d2 = torch.arange(d, device=path.device, dtype=torch.long) * (d * d)
        row_idx_d3_base = torch.arange(d * d, device=path.device, dtype=torch.long) * d

    # Storage for stream
    stream_out = []

    # Iterate over original steps
    for t in range(T):
        # We have gmax micro-steps to process
        # idx[:, t, :] is (B, gmax)
        # delta[:, t, :] is (B, gmax)

        step_idx = idx[:, t, :]
        step_delta = delta[:, t, :]

        # Process microsteps
        for k in range(gmax):
            j = step_idx[:, k]  # (B,)
            dlt = step_delta[:, k]  # (B,)

            # Mask out padded zeros (where delta is effectively 0)
            # Actually, standard scatter_add with 0 src adds 0, which is fine.
            # However, if j is garbage (e.g. from topk padding if we used -1), we need care.
            # topk padding usually repeats the last element or uses 0.
            # If dlt is 0, it doesn't matter where we add it, AS LONG AS index is valid.
            # topk indices are valid 0..d-1.

            # Update levels in reverse order (Chen product: S_new = S_old * exp(micro))
            # S_old is current levels.

            # --- Depth 3 Update ---
            if depth >= 3:
                # T3' = T3 + T2 x dlt + T1 x 0.5*dlt^2 + 1/6*dlt^3

                # 1. From T2: T3[:, :, j] += T2 * dlt
                # Indices in T3 (flat): T2_flat_idx * d + j
                # T2 is (B, d*d). We construct indices (B, d*d).
                # row_idx_d3_base is (d*d,) -> 0, d, 2d...
                # j is (B, 1) effectively

                # Expand j to match T2 size
                # shape (B, d*d)
                t3_indices_from_t2 = row_idx_d3_base[None, :] + j[:, None]
                t3_src_from_t2 = levels[1] * dlt[:, None]

                levels[2] = levels[2].scatter_add(1, t3_indices_from_t2, t3_src_from_t2)

                # 2. From T1: T3[:, j, j] += T1 * 0.5 * dlt^2
                # Indices in T3: T1_flat_idx * d*d + j*d + j
                # row_idx_d2 is (d,) -> 0, d^2, 2d^2...
                suffix = j * d + j  # (B,)
                t3_indices_from_t1 = row_idx_d2[None, :] + suffix[:, None]  # (B, d)

                val_sq_half = 0.5 * dlt * dlt
                t3_src_from_t1 = levels[0] * val_sq_half[:, None]  # (B, d)

                levels[2] = levels[2].scatter_add(1, t3_indices_from_t1, t3_src_from_t1)

                # 3. From Scalar: T3[j, j, j] += 1/6 * dlt^3
                # Index: j*d*d + j*d + j
                idx_scalar = j * (d * d) + j * d + j  # (B,)
                val_cube_sixth = (1.0 / 6.0) * dlt * dlt * dlt
                levels[2] = levels[2].scatter_add(
                    1, idx_scalar[:, None], val_cube_sixth[:, None]
                )

            # --- Depth 2 Update ---
            if depth >= 2:
                # T2' = T2 + T1 x dlt + 0.5*dlt^2

                # 1. From T1: T2[:, j] += T1 * dlt
                # Indices in T2: T1_idx * d + j
                # row_idx_d is (d,) -> 0, d, 2d...
                t2_indices_from_t1 = row_idx_d[None, :] + j[:, None]  # (B, d)
                t2_src_from_t1 = levels[0] * dlt[:, None]  # (B, d)

                levels[1] = levels[1].scatter_add(1, t2_indices_from_t1, t2_src_from_t1)

                # 2. From Scalar: T2[j, j] += 0.5 * dlt^2
                # Index: j*d + j
                idx_scalar_t2 = j * d + j
                val_sq_half = 0.5 * dlt * dlt
                levels[1] = levels[1].scatter_add(
                    1, idx_scalar_t2[:, None], val_sq_half[:, None]
                )

            # --- Depth 1 Update ---
            # T1' = T1 + dlt
            # Index: j
            levels[0] = levels[0].scatter_add(1, j[:, None], dlt[:, None])

        if stream:
            # Flatten levels into a single vector per batch item
            flat_sig = torch.cat([l.view(B, -1) for l in levels], dim=1)
            stream_out.append(flat_sig)

    if stream:
        return torch.stack(stream_out, dim=1)
    else:
        return torch.cat([l.view(B, -1) for l in levels], dim=1)
