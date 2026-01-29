"""
Correctness tests for all MoE FP8 block-scale kernel versions.

Compares each kernel version against the reference implementation (tests/reference.py).

Run:  python tests/test_kernel.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solution", "triton"))
sys.path.insert(0, os.path.dirname(__file__))

from kernel import kernel as kernel_v0, H, I, E_GLOBAL, E_LOCAL, BLOCK
from kernel_v1 import kernel as kernel_v1
from kernel_v2 import kernel as kernel_v2
from kernel_v3 import kernel as kernel_v3
from kernel_v4 import kernel as kernel_v4
from reference import run as reference_run

DEVICE = "cuda"

VERSIONS = {
    "v0_baseline": kernel_v0,
    "v1_fused_dequant": kernel_v1,
    "v2_grouped_gemm": kernel_v2,
    "v3_fused_swiglu": kernel_v3,
    "v4_token_permute": kernel_v4,
}

# Tolerance per version: V2+ use bf16 dot product so allow slightly more error
TOLERANCES = {
    "v0_baseline": 1e-2,
    "v1_fused_dequant": 5e-2,
    "v2_grouped_gemm": 5e-2,
    "v3_fused_swiglu": 5e-2,
    "v4_token_permute": 5e-2,
}


# =====================================================================
# Synthetic data generation
# =====================================================================

def make_fp8_with_block_scale_2d(shape, block_size=128, device=DEVICE):
    """Create a 2D FP8 tensor and its block-scale."""
    rows, cols = shape
    t = torch.randn(shape, device=device, dtype=torch.float32) * 0.1
    scale_shape = ((rows + block_size - 1) // block_size,
                   (cols + block_size - 1) // block_size)
    scale = torch.ones(scale_shape, device=device, dtype=torch.float32)
    fp8_t = torch.zeros(shape, device=device, dtype=torch.float8_e4m3fn)
    for rb in range(scale_shape[0]):
        for cb in range(scale_shape[1]):
            r_s, r_e = rb * block_size, min((rb + 1) * block_size, rows)
            c_s, c_e = cb * block_size, min((cb + 1) * block_size, cols)
            block_max = t[r_s:r_e, c_s:c_e].abs().max().item()
            if block_max > 0:
                scale[rb, cb] = block_max / 448.0
            fp8_t[r_s:r_e, c_s:c_e] = (t[r_s:r_e, c_s:c_e] / scale[rb, cb]).to(torch.float8_e4m3fn)
    return fp8_t, scale


def make_test_data(T, local_expert_offset=0, device=DEVICE):
    """Generate synthetic inputs matching expected shapes."""
    routing_logits = torch.randn(T, E_GLOBAL, device=device, dtype=torch.float32)
    routing_bias = torch.randn(E_GLOBAL, device=device, dtype=torch.float32) * 0.1

    # Hidden states: [T, H] FP8 with scale [H/128, T] (transposed layout)
    H_blocks = H // BLOCK
    hs_float = torch.randn(T, H, device=device, dtype=torch.float32) * 0.1
    hs_scale = torch.ones(H_blocks, T, device=device, dtype=torch.float32)
    hs_fp8 = torch.zeros(T, H, device=device, dtype=torch.float8_e4m3fn)
    for hb in range(H_blocks):
        h_s, h_e = hb * BLOCK, (hb + 1) * BLOCK
        for t in range(T):
            block_max = hs_float[t, h_s:h_e].abs().max().item()
            if block_max > 0:
                hs_scale[hb, t] = block_max / 448.0
            hs_fp8[t, h_s:h_e] = (hs_float[t, h_s:h_e] / hs_scale[hb, t]).to(torch.float8_e4m3fn)

    # GEMM1 weights: [E_LOCAL, 2*I, H]
    g1_fp8 = torch.zeros(E_LOCAL, 2 * I, H, device=device, dtype=torch.float8_e4m3fn)
    g1_scale = torch.ones(E_LOCAL, (2 * I) // BLOCK, H // BLOCK, device=device, dtype=torch.float32)
    for e in range(E_LOCAL):
        w, s = make_fp8_with_block_scale_2d((2 * I, H), device=device)
        g1_fp8[e] = w
        g1_scale[e] = s

    # GEMM2 weights: [E_LOCAL, H, I]
    g2_fp8 = torch.zeros(E_LOCAL, H, I, device=device, dtype=torch.float8_e4m3fn)
    g2_scale = torch.ones(E_LOCAL, H // BLOCK, I // BLOCK, device=device, dtype=torch.float32)
    for e in range(E_LOCAL):
        w, s = make_fp8_with_block_scale_2d((H, I), device=device)
        g2_fp8[e] = w
        g2_scale[e] = s

    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hs_fp8,
        "hidden_states_scale": hs_scale,
        "gemm1_weights": g1_fp8,
        "gemm1_weights_scale": g1_scale,
        "gemm2_weights": g2_fp8,
        "gemm2_weights_scale": g2_scale,
        "local_expert_offset": local_expert_offset,
        "routed_scaling_factor": 2.5,
    }


# =====================================================================
# Test functions (parameterized by kernel version)
# =====================================================================

def test_basic_correctness(fn, name, tol):
    """T=4: compare kernel output to reference."""
    print(f"  test_basic_correctness (T=4) ... ", end="", flush=True)
    torch.manual_seed(42)
    data = make_test_data(T=4, local_expert_offset=0)

    out_kernel = fn(**data)
    out_ref = reference_run(**data)

    assert out_kernel.shape == out_ref.shape == (4, H)
    assert out_kernel.dtype == torch.bfloat16

    max_abs_err = (out_kernel.float() - out_ref.float()).abs().max().item()
    ref_abs = out_ref.float().abs()
    nonzero = ref_abs > 1e-6
    if nonzero.any():
        max_rel_err = ((out_kernel.float() - out_ref.float()).abs()[nonzero] / ref_abs[nonzero]).max().item()
    else:
        max_rel_err = 0.0

    print(f"max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e} ... ", end="")
    assert max_abs_err < tol, f"Absolute error too large: {max_abs_err} (tol={tol})"
    print("PASSED")


def test_single_token(fn, name, tol):
    """T=1: compare kernel output to reference."""
    print(f"  test_single_token (T=1) ... ", end="", flush=True)
    torch.manual_seed(123)
    data = make_test_data(T=1, local_expert_offset=0)

    out_kernel = fn(**data)
    out_ref = reference_run(**data)

    assert out_kernel.shape == (1, H)
    max_abs_err = (out_kernel.float() - out_ref.float()).abs().max().item()
    print(f"max_abs_err={max_abs_err:.6e} ... ", end="")
    assert max_abs_err < tol, f"Absolute error too large: {max_abs_err} (tol={tol})"
    print("PASSED")


def test_no_local_experts(fn, name, tol):
    """All top-k experts fall outside local range -> output should be zero."""
    print(f"  test_no_local_experts ... ", end="", flush=True)
    torch.manual_seed(99)
    data = make_test_data(T=2, local_expert_offset=224)
    data["routing_logits"][:, 224:256] = -100.0
    data["routing_bias"][224:256] = -100.0

    out_kernel = fn(**data)
    out_ref = reference_run(**data)

    assert out_kernel.shape == (2, H)
    assert (out_ref == 0).all(), "Reference should be zero"
    assert (out_kernel == 0).all(), "Kernel should be zero"
    print("PASSED")


def test_output_shape_and_dtype(fn, name, tol):
    """Verify shape and dtype for various T values."""
    print(f"  test_output_shape_and_dtype ... ", end="", flush=True)
    torch.manual_seed(0)
    for T in [1, 2, 4]:
        data = make_test_data(T=T, local_expert_offset=0)
        out = fn(**data)
        assert out.shape == (T, H), f"Shape mismatch for T={T}: {out.shape}"
        assert out.dtype == torch.bfloat16, f"Dtype mismatch: {out.dtype}"
    print("PASSED")


def test_different_offsets(fn, name, tol):
    """Test with non-zero local_expert_offset."""
    print(f"  test_different_offsets ... ", end="", flush=True)
    torch.manual_seed(77)
    for offset in [0, 32, 128]:
        data = make_test_data(T=2, local_expert_offset=offset)
        out_kernel = fn(**data)
        out_ref = reference_run(**data)

        max_abs_err = (out_kernel.float() - out_ref.float()).abs().max().item()
        print(f"offset={offset} max_abs_err={max_abs_err:.6e} ", end="", flush=True)
        assert max_abs_err < tol, f"Error too large at offset={offset}: {max_abs_err} (tol={tol})"
    print("... PASSED")


def test_bitwise_match_reference(fn, name, tol):
    """Strict check: kernel output should closely match reference."""
    print(f"  test_bitwise_match_reference (T=2) ... ", end="", flush=True)
    torch.manual_seed(2024)
    data = make_test_data(T=2, local_expert_offset=0)

    out_kernel = fn(**data)
    out_ref = reference_run(**data)

    if torch.equal(out_kernel, out_ref):
        print("EXACT MATCH ... PASSED")
    else:
        max_abs_err = (out_kernel.float() - out_ref.float()).abs().max().item()
        print(f"not bitwise (max_abs_err={max_abs_err:.6e}), checking tolerance ... ", end="")
        assert max_abs_err < tol, f"Error too large: {max_abs_err} (tol={tol})"
        print("PASSED (within tolerance)")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    all_tests = [
        test_output_shape_and_dtype,
        test_no_local_experts,
        test_single_token,
        test_basic_correctness,
        test_different_offsets,
        test_bitwise_match_reference,
    ]

    failed = []

    for name, fn in VERSIONS.items():
        tol = TOLERANCES[name]
        print(f"\n{'='*60}")
        print(f"Testing {name} (tolerance={tol:.0e})")
        print(f"{'='*60}")
        for test_fn in all_tests:
            try:
                test_fn(fn, name, tol)
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(f"{name}/{test_fn.__name__}")
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILURES ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed for all versions!")
