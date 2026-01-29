"""
Performance benchmark: all kernel versions vs reference implementation.

Measures latency, throughput, and speedup across various batch sizes (T).

Run:  python tests/bench_kernel.py
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solution", "triton"))
sys.path.insert(0, os.path.dirname(__file__))

from kernel_v1 import kernel as kernel_v1, H, I, E_GLOBAL, E_LOCAL, BLOCK
from kernel_v2 import kernel as kernel_v2
from kernel_v3 import kernel as kernel_v3
from kernel_v4 import kernel as kernel_v4
from reference import run as reference_run

DEVICE = "cuda"
WARMUP_ITERS = 3
BENCH_ITERS = 10
T_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

VERSIONS = {
    "v1_fused_dequant": kernel_v1,
    "v2_grouped_gemm": kernel_v2,
    "v3_fused_swiglu": kernel_v3,
    "v4_token_permute": kernel_v4,
    "reference": reference_run,
}


def make_fp8_with_block_scale_2d(shape, block_size=128, device=DEVICE):
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


def make_bench_data(T, device=DEVICE):
    routing_logits = torch.randn(T, E_GLOBAL, device=device, dtype=torch.float32)
    routing_bias = torch.randn(E_GLOBAL, device=device, dtype=torch.float32) * 0.1

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

    g1_fp8 = torch.zeros(E_LOCAL, 2 * I, H, device=device, dtype=torch.float8_e4m3fn)
    g1_scale = torch.ones(E_LOCAL, (2 * I) // BLOCK, H // BLOCK, device=device, dtype=torch.float32)
    for e in range(E_LOCAL):
        w, s = make_fp8_with_block_scale_2d((2 * I, H), device=device)
        g1_fp8[e] = w
        g1_scale[e] = s

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
        "local_expert_offset": 0,
        "routed_scaling_factor": 2.5,
    }


def bench_fn(fn, data, warmup, iters, label):
    """Benchmark a function with CUDA synchronization. Returns median latency in ms."""
    for i in range(warmup):
        print(f"    {label} warmup {i+1}/{warmup}", flush=True)
        fn(**data)
        torch.cuda.synchronize()

    times = []
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(**data)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        times.append(elapsed_ms)
        print(f"    {label} iter {i+1}/{iters}: {elapsed_ms:.2f} ms", flush=True)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return {"median_ms": median, "mean_ms": mean, "min_ms": times[0], "max_ms": times[-1]}


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Warmup: {WARMUP_ITERS}, Benchmark iters: {BENCH_ITERS}")
    print(f"T values: {T_VALUES}")
    print(f"Versions: {list(VERSIONS.keys())}")
    print("=" * 120)

    # Collect all results: results[T][version_name] = stats
    all_results = {}

    for idx, T in enumerate(T_VALUES):
        print(f"\n[{idx+1}/{len(T_VALUES)}] Generating data for T={T} ...", flush=True)
        torch.manual_seed(42)
        data = make_bench_data(T)
        torch.cuda.synchronize()
        print(f"  Data ready. Starting benchmarks.", flush=True)

        t_results = {}

        for v_name, v_fn in VERSIONS.items():
            print(f"  Benchmarking {v_name} (T={T}) ...", flush=True)
            try:
                stats = bench_fn(v_fn, data, WARMUP_ITERS, BENCH_ITERS, v_name)
                t_results[v_name] = stats
                print(f"  => {v_name}: {stats['median_ms']:.2f} ms (median)", flush=True)
            except Exception as e:
                print(f"  => {v_name}: FAILED ({e})", flush=True)
                t_results[v_name] = None

        all_results[T] = t_results

        del data
        torch.cuda.empty_cache()

    # Summary table
    version_names = list(VERSIONS.keys())
    col_width = 16

    print("\n" + "=" * 120)
    print("SUMMARY (median latency in ms)")
    print("=" * 120)

    # Header
    header = f"{'T':>6}"
    for v in version_names:
        header += f" | {v:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for T in T_VALUES:
        row = f"{T:>6}"
        for v in version_names:
            stats = all_results.get(T, {}).get(v)
            if stats is None:
                row += f" | {'FAILED':>{col_width}}"
            else:
                row += f" | {stats['median_ms']:>{col_width}.2f}"
        print(row)

    # Speedup table (vs reference)
    ref_name = "reference"
    print(f"\n{'='*120}")
    print(f"SPEEDUP vs {ref_name}")
    print("=" * 120)

    header = f"{'T':>6}"
    for v in version_names:
        if v == ref_name:
            continue
        header += f" | {v:>{col_width}}"
    print(header)
    print("-" * len(header))

    for T in T_VALUES:
        ref_stats = all_results.get(T, {}).get(ref_name)
        if ref_stats is None:
            continue
        row = f"{T:>6}"
        for v in version_names:
            if v == ref_name:
                continue
            stats = all_results.get(T, {}).get(v)
            if stats is None:
                row += f" | {'N/A':>{col_width}}"
            else:
                speedup = ref_stats["median_ms"] / stats["median_ms"]
                row += f" | {speedup:>{col_width}.2f}x"
        print(row)

    print("=" * 120)


if __name__ == "__main__":
    main()
