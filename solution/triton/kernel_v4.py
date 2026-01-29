"""
V4: Token Permute/Unpermute optimization.

Optimization: Vectorized token permutation via argsort (no per-expert Python loop
for building sorted lists). Triton weighted scatter-add kernel for unpermutation
replaces PyTorch index_add_. Built on V3 (grouped GEMM + Triton SwiGLU).
"""

import torch
import triton
import triton.language as tl

# ----- constants -----
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128


# =====================================================================
# 1. Triton Grouped GEMM Kernel (same as V2/V3)
# =====================================================================

@triton.jit
def grouped_gemm_kernel(
    A_ptr, A_scale_ptr,
    B_ptr, B_scale_ptr,
    C_ptr,
    tile_expert_ptr, tile_m_start_ptr, tile_m_count_ptr,
    N, K: tl.constexpr, K_blocks,
    stride_a_row, stride_a_col,
    stride_as_kb, stride_as_tok,
    stride_b_e, stride_b_n, stride_b_k,
    stride_bs_e, stride_bs_nb, stride_bs_kb,
    stride_c_row, stride_c_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_IS_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_idx = tl.load(tile_expert_ptr + pid_m)
    m_start = tl.load(tile_m_start_ptr + pid_m)
    m_count = tl.load(tile_m_count_ptr + pid_m)

    n_start = pid_n * BLOCK_N
    m_range = tl.arange(0, BLOCK_M)
    n_range = tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)

    m_mask = m_range < m_count
    n_mask = (n_start + n_range) < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K_blocks):
        k_start = kb * BLOCK_K

        a_offsets = (m_start + m_range)[:, None] * stride_a_row + (k_start + k_range)[None, :] * stride_a_col
        a_mask = m_mask[:, None] & (True)
        a_vals = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        if A_IS_FP8:
            a_float = a_vals.to(tl.float32)
            a_scale_offsets = kb * stride_as_kb + (m_start + m_range) * stride_as_tok
            a_scales = tl.load(A_scale_ptr + a_scale_offsets, mask=m_mask, other=1.0)
            a_scaled = a_float * a_scales[:, None]
        else:
            a_scaled = a_vals

        b_offsets = (expert_idx * stride_b_e
                     + (n_start + n_range)[None, :] * stride_b_n
                     + (k_start + k_range)[:, None] * stride_b_k)
        b_mask = n_mask[None, :] & (True)
        b_fp8 = tl.load(B_ptr + b_offsets, mask=b_mask, other=0.0)
        b_float = b_fp8.to(tl.float32)

        nb_val = n_start // BLOCK_K
        b_scale = tl.load(B_scale_ptr + expert_idx * stride_bs_e + nb_val * stride_bs_nb + kb * stride_bs_kb)
        b_scaled = b_float * b_scale

        acc += tl.dot(a_scaled.to(tl.bfloat16), b_scaled.to(tl.bfloat16))

    c_offsets = (m_start + m_range)[:, None] * stride_c_row + (n_start + n_range)[None, :] * stride_c_col
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptr + c_offsets, acc, mask=c_mask)


# =====================================================================
# 2. Triton SwiGLU Kernel (same as V3)
# =====================================================================

@triton.jit
def swiglu_kernel(
    input_ptr, output_ptr,
    total_tokens, half_size,
    stride_in_row, stride_in_col,
    stride_out_row, stride_out_col,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    if pid_row >= total_tokens:
        return

    col_start = pid_col * BLOCK_COLS
    col_range = col_start + tl.arange(0, BLOCK_COLS)
    col_mask = col_range < half_size

    x1_offsets = pid_row * stride_in_row + col_range * stride_in_col
    x2_offsets = pid_row * stride_in_row + (col_range + half_size) * stride_in_col

    x1 = tl.load(input_ptr + x1_offsets, mask=col_mask, other=0.0)
    x2 = tl.load(input_ptr + x2_offsets, mask=col_mask, other=0.0)

    silu_x2 = x2 * tl.sigmoid(x2)
    result = silu_x2 * x1

    out_offsets = pid_row * stride_out_row + col_range * stride_out_col
    tl.store(output_ptr + out_offsets, result, mask=col_mask)


def triton_swiglu(input_tensor, half_size):
    total = input_tensor.shape[0]
    output = torch.empty(total, half_size, dtype=input_tensor.dtype, device=input_tensor.device)
    BLOCK_COLS = 256
    grid = (total, triton.cdiv(half_size, BLOCK_COLS))
    swiglu_kernel[grid](
        input_tensor, output,
        total, half_size,
        input_tensor.stride(0), input_tensor.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_COLS=BLOCK_COLS,
    )
    return output


# =====================================================================
# 3. Triton Weighted Scatter-Add Kernel
# =====================================================================

@triton.jit
def weighted_scatter_add_kernel(
    src_ptr,          # [total, hidden] fp32 - expert outputs
    dst_ptr,          # [T, hidden] fp32 - output buffer
    weights_ptr,      # [total] fp32 - routing weights per pair
    token_ids_ptr,    # [total] int64 - original token indices
    total,
    hidden,
    stride_src_row, stride_src_col,
    stride_dst_row, stride_dst_col,
    BLOCK_H: tl.constexpr,
):
    """Weighted scatter-add: dst[token_id] += src[row] * weight[row]."""
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    if pid_row >= total:
        return

    col_start = pid_col * BLOCK_H
    col_range = col_start + tl.arange(0, BLOCK_H)
    col_mask = col_range < hidden

    # Load source values
    src_offsets = pid_row * stride_src_row + col_range * stride_src_col
    src_vals = tl.load(src_ptr + src_offsets, mask=col_mask, other=0.0)

    # Load weight and token id
    w = tl.load(weights_ptr + pid_row)
    tok_id = tl.load(token_ids_ptr + pid_row)

    weighted = src_vals * w

    # Atomic add to destination
    dst_offsets = tok_id * stride_dst_row + col_range * stride_dst_col
    tl.atomic_add(dst_ptr + dst_offsets, weighted, mask=col_mask)


def triton_weighted_scatter_add(src, dst, weights, token_ids):
    """Apply weighted scatter-add using Triton kernel."""
    total = src.shape[0]
    hidden = src.shape[1]
    BLOCK_H = 256
    grid = (total, triton.cdiv(hidden, BLOCK_H))
    weighted_scatter_add_kernel[grid](
        src, dst, weights, token_ids,
        total, hidden,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_H=BLOCK_H,
    )


# =====================================================================
# 4. Vectorized token permutation (no per-expert Python loop)
# =====================================================================

def build_sorted_tokens_vectorized(topk_idx, local_start, T, device):
    """Build sorted token-expert pairs using vectorized operations.

    Instead of looping over each expert to find its tokens, this:
    1. Creates all (token, expert) pairs from topk_idx
    2. Filters to local experts
    3. Sorts by expert using argsort for contiguous memory access

    Returns:
        sorted_token_ids: [total_pairs] int64
        sorted_expert_local: [total_pairs] int64
        sorted_expert_global: [total_pairs] int64
        expert_offsets: [E_LOCAL+1] int64 tensor
    """
    # Create all (token, expert) pairs: [T, TOP_K]
    token_indices = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K)
    expert_global = topk_idx  # [T, TOP_K]

    # Flatten
    flat_tok = token_indices.reshape(-1)     # [T*TOP_K]
    flat_exp = expert_global.reshape(-1)     # [T*TOP_K]

    # Filter to local experts
    local_end = local_start + E_LOCAL
    local_mask = (flat_exp >= local_start) & (flat_exp < local_end)
    local_tok = flat_tok[local_mask]
    local_exp_global = flat_exp[local_mask]
    local_exp_local = local_exp_global - local_start

    total_pairs = local_tok.numel()
    if total_pairs == 0:
        expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
        return (torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
                expert_offsets)

    # Sort by local expert for contiguous memory access
    sort_order = torch.argsort(local_exp_local, stable=True)
    sorted_token_ids = local_tok[sort_order]
    sorted_expert_local = local_exp_local[sort_order]
    sorted_expert_global = local_exp_global[sort_order]

    # Compute expert offsets using bincount
    expert_counts = torch.zeros(E_LOCAL, dtype=torch.int64, device=device)
    counts = torch.bincount(sorted_expert_local, minlength=E_LOCAL)
    expert_counts[:] = counts[:E_LOCAL]
    expert_offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, 0)

    return sorted_token_ids, sorted_expert_local, sorted_expert_global, expert_offsets


# =====================================================================
# 5. Helper: build tile mapping (same as V2/V3)
# =====================================================================

def build_tile_mapping(expert_offsets, num_experts, BLOCK_M, device):
    tile_expert_list = []
    tile_m_start_list = []
    tile_m_count_list = []

    for e_idx in range(num_experts):
        start = int(expert_offsets[e_idx])
        end = int(expert_offsets[e_idx + 1])
        n_tokens = end - start
        if n_tokens == 0:
            continue
        n_tiles = (n_tokens + BLOCK_M - 1) // BLOCK_M
        for t in range(n_tiles):
            tile_expert_list.append(e_idx)
            tile_m_start_list.append(start + t * BLOCK_M)
            remaining = n_tokens - t * BLOCK_M
            tile_m_count_list.append(min(remaining, BLOCK_M))

    if len(tile_expert_list) == 0:
        return (torch.zeros(1, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
                0)

    tile_expert = torch.tensor(tile_expert_list, dtype=torch.int32, device=device)
    tile_m_start = torch.tensor(tile_m_start_list, dtype=torch.int32, device=device)
    tile_m_count = torch.tensor(tile_m_count_list, dtype=torch.int32, device=device)
    return tile_expert, tile_m_start, tile_m_count, len(tile_expert_list)


# =====================================================================
# 6. Main kernel entry point
# =====================================================================

def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    V4: Grouped GEMM + Triton SwiGLU + vectorized permute + Triton scatter-add.
    Returns: [T, H] bfloat16 output tensor.
    """
    T = routing_logits.shape[0]
    device = routing_logits.device

    # ---- 1) No-aux routing ----

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    group_size = E_GLOBAL // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M_mask = torch.zeros_like(s)
    M_mask.scatter_(1, topk_idx, 1.0)
    weights = s * M_mask
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # ---- 2) Vectorized token permutation ----

    local_start = int(local_expert_offset)
    sorted_token_ids, sorted_expert_local, sorted_expert_global, expert_offsets = \
        build_sorted_tokens_vectorized(topk_idx, local_start, T, device)

    total_pairs = sorted_token_ids.numel()
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    if total_pairs == 0:
        return output.to(torch.bfloat16)

    # ---- 3) Gather sorted inputs ----

    A_sorted = hidden_states[sorted_token_ids].contiguous()
    A_scale_sorted = hidden_states_scale[:, sorted_token_ids].contiguous()

    # ---- 4) Grouped GEMM1 ----

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    N1 = 2 * I
    K1 = H
    K1_blocks = K1 // BLOCK_K

    tile_expert, tile_m_start, tile_m_count, total_m_tiles = \
        build_tile_mapping(expert_offsets, E_LOCAL, BLOCK_M, device)

    gemm1_out = torch.zeros((total_pairs, N1), dtype=torch.float32, device=device)

    if total_m_tiles > 0:
        n1_tiles = triton.cdiv(N1, BLOCK_N)
        grouped_gemm_kernel[(total_m_tiles, n1_tiles)](
            A_sorted, A_scale_sorted,
            gemm1_weights, gemm1_weights_scale,
            gemm1_out,
            tile_expert, tile_m_start, tile_m_count,
            N1, K1, K1_blocks,
            A_sorted.stride(0), A_sorted.stride(1),
            A_scale_sorted.stride(0), A_scale_sorted.stride(1),
            gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
            gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
            gemm1_out.stride(0), gemm1_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            A_IS_FP8=True,
        )

    # ---- 5) Triton SwiGLU ----

    intermediate = triton_swiglu(gemm1_out, I)

    # ---- 6) Grouped GEMM2 ----

    N2 = H
    K2 = I
    K2_blocks = K2 // BLOCK_K

    gemm2_out = torch.zeros((total_pairs, N2), dtype=torch.float32, device=device)

    if total_m_tiles > 0:
        n2_tiles = triton.cdiv(N2, BLOCK_N)
        dummy_scale = torch.ones(1, dtype=torch.float32, device=device)
        grouped_gemm_kernel[(total_m_tiles, n2_tiles)](
            intermediate, dummy_scale,
            gemm2_weights, gemm2_weights_scale,
            gemm2_out,
            tile_expert, tile_m_start, tile_m_count,
            N2, K2, K2_blocks,
            intermediate.stride(0), intermediate.stride(1),
            0, 0,
            gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
            gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
            gemm2_out.stride(0), gemm2_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            A_IS_FP8=False,
        )

    # ---- 7) Triton weighted scatter-add ----

    pair_weights = weights[sorted_token_ids, sorted_expert_global]
    triton_weighted_scatter_add(gemm2_out, output, pair_weights, sorted_token_ids)

    return output.to(torch.bfloat16)
