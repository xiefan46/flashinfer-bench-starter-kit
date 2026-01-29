"""
V1: Fused Dequant - eliminates repeat_interleave intermediate tensors.

Optimization: 3D Triton dequant kernel for weights avoids materializing
huge expanded scale tensors. Outputs bf16 instead of fp32 to reduce memory.
Expert loop and matmul still use PyTorch.
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
# 1. FP8 Block-Scale Dequantization Kernels
# =====================================================================

@triton.jit
def dequant_fp8_block_2d_kernel(
    fp8_ptr, scale_ptr, out_ptr,
    rows, cols,
    stride_fp8_row, stride_fp8_col,
    stride_scale_row, stride_scale_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Dequantize a 2D FP8 tensor with block-scale to BF16."""
    pid = tl.program_id(0)
    num_col_tiles = tl.cdiv(cols, TILE_COLS)
    tile_row = pid // num_col_tiles
    tile_col = pid % num_col_tiles

    row_start = tile_row * TILE_ROWS
    col_start = tile_col * TILE_COLS

    rows_offsets = row_start + tl.arange(0, TILE_ROWS)
    cols_offsets = col_start + tl.arange(0, TILE_COLS)

    row_mask = rows_offsets < rows
    col_mask = cols_offsets < cols
    mask = row_mask[:, None] & col_mask[None, :]

    fp8_offsets = rows_offsets[:, None] * stride_fp8_row + cols_offsets[None, :] * stride_fp8_col
    fp8_vals = tl.load(fp8_ptr + fp8_offsets, mask=mask, other=0.0)
    fp8_float = fp8_vals.to(tl.float32)

    scale_row_idx = rows_offsets // BLOCK_SIZE
    scale_col_idx = cols_offsets // BLOCK_SIZE
    scale_offsets = scale_row_idx[:, None] * stride_scale_row + scale_col_idx[None, :] * stride_scale_col
    scales = tl.load(scale_ptr + scale_offsets, mask=mask, other=1.0)

    result = (fp8_float * scales).to(tl.bfloat16)
    out_offsets = rows_offsets[:, None] * stride_out_row + cols_offsets[None, :] * stride_out_col
    tl.store(out_ptr + out_offsets, result, mask=mask)


def dequant_fp8_block_2d(fp8_tensor, scale_tensor, rows, cols, scale_row_stride, scale_col_stride):
    """Dequantize 2D FP8 tensor. Returns BF16 tensor [rows, cols]."""
    out = torch.empty(rows, cols, dtype=torch.bfloat16, device=fp8_tensor.device)
    TILE = 128
    grid = (triton.cdiv(rows, TILE) * triton.cdiv(cols, TILE),)
    dequant_fp8_block_2d_kernel[grid](
        fp8_tensor, scale_tensor, out,
        rows, cols,
        fp8_tensor.stride(0), fp8_tensor.stride(1),
        scale_row_stride, scale_col_stride,
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK,
        TILE_ROWS=TILE,
        TILE_COLS=TILE,
    )
    return out


@triton.jit
def dequant_fp8_block_3d_kernel(
    fp8_ptr, scale_ptr, out_ptr,
    E, rows, cols,
    stride_fp8_e, stride_fp8_row, stride_fp8_col,
    stride_scale_e, stride_scale_row, stride_scale_col,
    stride_out_e, stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Dequantize a 3D FP8 tensor [E, rows, cols] with block-scale to BF16."""
    pid = tl.program_id(0)
    num_col_tiles = tl.cdiv(cols, TILE_COLS)
    num_row_tiles = tl.cdiv(rows, TILE_ROWS)
    tiles_per_expert = num_row_tiles * num_col_tiles

    expert_id = pid // tiles_per_expert
    local_pid = pid % tiles_per_expert
    tile_row = local_pid // num_col_tiles
    tile_col = local_pid % num_col_tiles

    row_start = tile_row * TILE_ROWS
    col_start = tile_col * TILE_COLS

    rows_offsets = row_start + tl.arange(0, TILE_ROWS)
    cols_offsets = col_start + tl.arange(0, TILE_COLS)

    row_mask = rows_offsets < rows
    col_mask = cols_offsets < cols
    mask = row_mask[:, None] & col_mask[None, :]

    fp8_offsets = (expert_id * stride_fp8_e
                   + rows_offsets[:, None] * stride_fp8_row
                   + cols_offsets[None, :] * stride_fp8_col)
    fp8_vals = tl.load(fp8_ptr + fp8_offsets, mask=mask, other=0.0)
    fp8_float = fp8_vals.to(tl.float32)

    scale_row_idx = rows_offsets // BLOCK_SIZE
    scale_col_idx = cols_offsets // BLOCK_SIZE
    scale_offsets = (expert_id * stride_scale_e
                     + scale_row_idx[:, None] * stride_scale_row
                     + scale_col_idx[None, :] * stride_scale_col)
    scales = tl.load(scale_ptr + scale_offsets, mask=mask, other=1.0)

    result = (fp8_float * scales).to(tl.bfloat16)
    out_offsets = (expert_id * stride_out_e
                   + rows_offsets[:, None] * stride_out_row
                   + cols_offsets[None, :] * stride_out_col)
    tl.store(out_ptr + out_offsets, result, mask=mask)


def dequant_fp8_block_3d(fp8_tensor, scale_tensor):
    """Dequantize 3D FP8 tensor [E, rows, cols]. Returns BF16 tensor."""
    E, rows, cols = fp8_tensor.shape
    out = torch.empty(E, rows, cols, dtype=torch.bfloat16, device=fp8_tensor.device)
    TILE = 128
    tiles_per_expert = triton.cdiv(rows, TILE) * triton.cdiv(cols, TILE)
    grid = (E * tiles_per_expert,)
    dequant_fp8_block_3d_kernel[grid](
        fp8_tensor, scale_tensor, out,
        E, rows, cols,
        fp8_tensor.stride(0), fp8_tensor.stride(1), fp8_tensor.stride(2),
        scale_tensor.stride(0), scale_tensor.stride(1), scale_tensor.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE=BLOCK,
        TILE_ROWS=TILE,
        TILE_COLS=TILE,
    )
    return out


# =====================================================================
# 2. Main kernel entry point
# =====================================================================

def kernel(
    routing_logits,        # [T, E_global]     float32
    routing_bias,          # [E_global]         float32
    hidden_states,         # [T, H]            float8_e4m3fn
    hidden_states_scale,   # [H/128, T]        float32  (transposed)
    gemm1_weights,         # [E_local, 2I, H]  float8_e4m3fn
    gemm1_weights_scale,   # [E_local, 2I/128, H/128] float32
    gemm2_weights,         # [E_local, H, I]   float8_e4m3fn
    gemm2_weights_scale,   # [E_local, H/128, I/128]  float32
    local_expert_offset,   # int
    routed_scaling_factor, # float
):
    """
    V1: Fused dequant via Triton 3D kernel. No repeat_interleave.
    Returns: [T, H] bfloat16 output tensor.
    """
    T = routing_logits.shape[0]
    device = routing_logits.device

    # ---- 1) FP8 block-scale dequantization (fused, no repeat_interleave) ----

    # Hidden states: [T, H], scale: [H/128, T] → transposed to [T, H/128]
    A_scale_TH = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()
    A = dequant_fp8_block_2d(hidden_states, A_scale_TH, T, H,
                             A_scale_TH.stride(0), A_scale_TH.stride(1))

    # W13: [E_LOCAL, 2I, H] → bf16 via 3D Triton dequant
    W13 = dequant_fp8_block_3d(gemm1_weights, gemm1_weights_scale)

    # W2: [E_LOCAL, H, I] → bf16 via 3D Triton dequant
    W2 = dequant_fp8_block_3d(gemm2_weights, gemm2_weights_scale)

    # ---- 2) No-aux routing ----

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

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # ---- 3) Local expert compute (PyTorch matmul, bf16 weights upcast) ----

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    A_f32 = A.float()

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        sel_mask_per_token = (topk_idx == ge).any(dim=1)
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)

        A_e = A_f32.index_select(0, token_idx)
        W13_e = W13[le].float()
        W2_e = W2[le].float()

        # GEMM1
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1

        # GEMM2
        O = C.matmul(W2_e.t())

        # Weighted accumulation
        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)
