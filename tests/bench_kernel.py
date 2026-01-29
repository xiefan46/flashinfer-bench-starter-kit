"""
Performance benchmark: kernel vs reference implementation.

Measures latency, throughput, and speedup across various batch sizes (T).

Run:  python tests/bench_kernel.py
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solution", "triton"))
sys.path.insert(0, os.path.dirname(__file__))

from kernel import kernel, H, I, E_GLOBAL, E_LOCAL, BLOCK
from reference import run as reference_run

DEVICE = "cuda"
WARMUP_ITERS = 3
BENCH_ITERS = 10
T_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


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
    # Warmup
    for i in range(warmup):
        print(f"    {label} warmup {i+1}/{warmup}", flush=True)
        fn(**data)
        torch.cuda.synchronize()

    # Benchmark
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
    print("=" * 90)

    results = []

    for idx, T in enumerate(T_VALUES):
        print(f"\n[{idx+1}/{len(T_VALUES)}] Generating data for T={T} ...", flush=True)
        torch.manual_seed(42)
        data = make_bench_data(T)
        torch.cuda.synchronize()
        print(f"  Data ready. Starting benchmark.", flush=True)

        # Benchmark kernel
        print(f"  Benchmarking kernel (T={T}) ...", flush=True)
        kernel_stats = bench_fn(kernel, data, WARMUP_ITERS, BENCH_ITERS, "kernel")

        # Benchmark reference
        print(f"  Benchmarking reference (T={T}) ...", flush=True)
        ref_stats = bench_fn(reference_run, data, WARMUP_ITERS, BENCH_ITERS, "reference")

        speedup = ref_stats["median_ms"] / kernel_stats["median_ms"] if kernel_stats["median_ms"] > 0 else float("inf")
        kernel_tps = T / (kernel_stats["median_ms"] / 1000)
        ref_tps = T / (ref_stats["median_ms"] / 1000)

        results.append({
            "T": T,
            "kernel_median_ms": kernel_stats["median_ms"],
            "kernel_mean_ms": kernel_stats["mean_ms"],
            "ref_median_ms": ref_stats["median_ms"],
            "ref_mean_ms": ref_stats["mean_ms"],
            "speedup": speedup,
            "kernel_tps": kernel_tps,
            "ref_tps": ref_tps,
        })

        print(f"  => kernel: {kernel_stats['median_ms']:.2f} ms (median), "
              f"reference: {ref_stats['median_ms']:.2f} ms (median), "
              f"speedup: {speedup:.2f}x", flush=True)

        # Free data to reclaim memory
        del data
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'T':>6} | {'Kernel (ms)':>12} | {'Reference (ms)':>14} | {'Speedup':>8} | {'Kernel TPS':>12} | {'Ref TPS':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['T']:>6} | {r['kernel_median_ms']:>12.2f} | {r['ref_median_ms']:>14.2f} | "
              f"{r['speedup']:>7.2f}x | {r['kernel_tps']:>12.1f} | {r['ref_tps']:>12.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
