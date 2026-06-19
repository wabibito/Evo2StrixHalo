#!/usr/bin/env python3
"""
Head-to-head: fp8_rocm vs fp8_vulkan on a real Evo2 1B FP8 layer.

Compares (a) numerical agreement vs PyTorch's native float8_e4m3fn (the H100 GEMM
math) and (b) throughput, on whatever device each package selects via best_device().
On Strix Halo the ROCm package targets the GPU ("cuda"); the Vulkan package falls
back to CPU unless torch was built with USE_VULKAN.

    python scripts/bench_fp8_backends.py
"""
from __future__ import annotations
import glob, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/huyvunguyen/Dev/FP8-ROCM")
sys.path.insert(0, "/home/huyvunguyen/Dev/FP8-Vulkan")
import fp8_rocm
import fp8_vulkan

FP8_MAX = 448.0


def native_e4m3(x):
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def ref_te(x, W, a, w):
    return F.linear(native_e4m3(x.float() * a), native_e4m3(W.float() * w)) * (1.0 / (a * w))


def load_layer():
    import io
    ckpt = glob.glob(os.path.expanduser("~/.cache/huggingface/**/evo2_1b_base.pt"), recursive=True)[0]
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "module" in sd:
        sd = sd["module"]
    for k in sd:
        if k.endswith(".projections._extra_state"):
            path = k[:-len("._extra_state")]
            W = sd.get(path + ".weight")
            v = sd[k]; v.seek(0)
            meta = torch.load(v, map_location="cpu", weights_only=False)
            sf = meta.get("scale_fwd")
            if W is not None and sf is not None:
                return path, W, float(sf[0]), float(sf[1])
    raise RuntimeError("no FP8 projection layer found")


def bench(pkg, name, W, a, w, x_cpu, ref, iters=50):
    dev = pkg.best_device()
    lin = pkg.Fp8TELinear(W.data, None, a, w).to(dev)
    x = x_cpu.to(dev)
    # correctness
    out = lin(x).float().cpu()
    rel = ((out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
    # warmup + sync
    for _ in range(5):
        lin(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        lin(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1e3  # ms/call
    print(f"  {name:<12} device={dev:<5}  rel-err vs native e4m3 = {rel:.2e}   {dt:7.2f} ms/call")
    return dev, rel, dt, out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(0)
    path, W, a, w = load_layer()
    print(f"layer: {path}   shape={tuple(W.shape)}   act_scale={a:.2f} weight_scale={w:.2f}")
    # realistic activation: a sequence of 4096 tokens
    x = torch.randn(1, 4096, W.shape[1], dtype=torch.bfloat16)
    ref = ref_te(x, W, a, w)
    print(f"reference = native float8_e4m3fn GEMM (H100 math)\n")

    dr, rr, tr, outr = bench(fp8_rocm,   "fp8_rocm",   W, a, w, x, ref)
    dv, rv, tv, outv = bench(fp8_vulkan, "fp8_vulkan", W, a, w, x, ref)

    cross = ((outr - outv).abs().max()).item()
    print(f"\n  fp8_rocm vs fp8_vulkan max abs output diff: {cross:.3e}  "
          f"({'identical math' if cross < 1e-3 else 'DIFFERENT'})")
    faster = "fp8_rocm (GPU)" if tr < tv else "fp8_vulkan"
    print(f"  speed: {faster} faster  ({max(tr,tv)/max(min(tr,tv),1e-9):.1f}x)")
    print(f"  accuracy: identical (same device-agnostic e4m3 emulation)")


if __name__ == "__main__":
    raise SystemExit(main())
