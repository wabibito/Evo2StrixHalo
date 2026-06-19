# Evo2StrixHalo

A port of [Evo 2](https://github.com/arcinstitute/evo2) (Arc Institute's DNA
language model) to **AMD Strix Halo** — the Ryzen AI Max+ 395 (Radeon 8060S,
`gfx1151`) — running on **ROCm** under Linux.

> **Status: new and unverified on hardware.** This port is built
> correct-by-construction from the upstream code plus the ROCm/Strix Halo
> facts, but it has not yet been run end-to-end on a 395. Expect to iterate.
> The `evo2` Python package, the FP8 emulation, and the test scripts are
> ported from the validated [Evo2MPS](https://github.com/wabibito/Evo2MPS)
> work; the device, install, and patch layers are ROCm-specific.

This is a fork of [arcinstitute/evo2](https://github.com/arcinstitute/evo2)
(`upstream` remote) with edits to device handling, the FP8 fallback, and the
runtime patches so Evo 2 runs on Strix Halo without an NVIDIA GPU. Upstream
documentation is preserved in [`README.upstream.md`](README.upstream.md).

## What works (and what doesn't) on Strix Halo

Strix Halo's iGPU is **RDNA 3.5**, which has **no FP8 tensor-core hardware**.
Transformer Engine (the FP8 path Evo 2's larger checkpoints were trained with)
requires CDNA3/MI300 on AMD, or NVIDIA Ada/Hopper. So:

| Checkpoint | On Strix Halo |
|-----------|---------------|
| `evo2_7b`, `evo2_7b_base` (8K) | **bf16-native — the accurate workhorse.** RDNA 3.5 runs bf16 well, and these ship with FP8 off. |
| `evo2_1b_base` | FP8-trained; loads with **e4m3 emulation** of the input projections (recovers most accuracy lost to bf16). |
| `evo2_20b` | **Loads and runs, but not numerically usable** (~24% next-token acc vs the 91.7% H100 ref — chance for 4-base DNA). FP8 across 117 layers; emulating the linears (any precision/scale, and a CPU native-`float8_e4m3fn` reference) leaves it at chance, so its FP8 dependence is in the attention/fused stack, not the linears. Not recoverable by emulation. |
| `evo2_40b` | Same numerical outcome as the 20B, and additionally storage-bound on a consumer device (sharded checkpoint merges to ~80 GB, ~165 GB peak working disk). |

This is the same FP8 hardware boundary that affects Apple Silicon — it is not
specific to AMD.

## Requirements

- AMD Strix Halo (Ryzen AI Max+ 395 / `gfx1151`), or another RDNA 3/3.5 GPU
- **Ubuntu 26.04** (baseline), with **ROCm 7.x** installed
  ([ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/))
- Python 3.11+

`gfx1151` is not on AMD's official ROCm support matrix as of early 2026 but
works in practice with ROCm 7.x. If the GPU isn't detected, set
`export HSA_OVERRIDE_GFX_VERSION=11.5.1`.

## Quick start

```bash
git clone https://github.com/wabibito/Evo2StrixHalo.git
cd Evo2StrixHalo
./install.sh                      # ROCm wheels + vtx + patches
source .venv/bin/activate

# Smoke test (downloads ~4 GB on first run):
python scripts/smoke_test.py --model evo2_1b_base

# Full DNA pipeline on the accurate 7B:
python scripts/test_dna.py --model evo2_7b_base

# Verify everything against reference data:
python scripts/verify_all.py
```

`install.sh` creates a venv, installs the `gfx1151` PyTorch ROCm wheels
(`https://rocm.nightlies.amd.com/v2/gfx1151/`, overridable via
`EVO2_TORCH_INDEX`), installs `vtx` and this package, then applies the vortex
patches. The patch script auto-detects ROCm and applies only the flash-attn
fallbacks (gfx1151 has no flash-attn); the broader MPS device-reroute patches
are skipped because ROCm presents through the CUDA API natively.

## FP8 emulation

The 1B (and, in principle, the 20B/40B) are loaded with bf16 projections and
then have NVIDIA Transformer Engine's per-tensor e4m3 input-projection GEMM
**emulated** in pure PyTorch. `evo2/fp8_emulation.py` recovers the per-layer
scales from the checkpoint's TE `_extra_state` and delegates the linear-layer
emulation to the [FP8-ROCM](https://github.com/wabibito/FP8-ROCM) package
(`Fp8TELinear`, bf16 matrix-core GEMM), falling back to an in-repo
implementation if FP8-ROCM is not installed. Applied automatically for
`evo2_1b_base`; control it with `EVO2_FP8_EMULATION=0/1`.

**Validated on Strix Halo (Radeon 8060S, gfx1151, ROCm 7.2.4):**

| Model | Path | Next-token acc | vs H100 ref |
|---|---|---|---|
| `evo2_7b_base` | bf16-native | 85.98% | +0.06 pp (essentially exact) |
| `evo2_1b_base` | bf16 fallback (no FP8) | 32.57% | -46.99 pp |
| `evo2_1b_base` | **e4m3 emulation (bf16)** | **78.68%** | **-0.87 pp** |

The consistent-bf16 emulation matches the operator the checkpoint was calibrated
against (vortex's bf16 forward) and runs on the iGPU's matrix cores, so it is
both faster (~15x on the projection GEMM) and ~10 pp more accurate than a
mixed-precision or fp32 emulation. `torch._scaled_mm` is unavailable on gfx1151
(MI300+/Hopper only), and RDNA 3.5 has no FP8 WMMA units (those are RDNA 4), so
emulation is the only path. See the
[FP8-ROCM technical paper](https://github.com/wabibito/FP8-ROCM/blob/main/docs/PAPER.md)
for the full study.

Emulation does **not** rescue the 20B/40B. The 20B is FP8 across 117 layers, and
emulating those linears leaves it at chance (~24% vs 91.7%) under every
configuration tested — bf16/fp32 grid, bf16/fp32 accumulation,
checkpoint/dynamic activation scales, and any layer subset. The decisive control:
running the whole 20B on **CPU** with PyTorch's **native `float8_e4m3fn`** cast
(the exact FP8 arithmetic an H100 performs) also gives chance, so it is neither a
ROCm artifact nor an emulation-fidelity gap. The 20B's FP8 dependence lives in
the attention/fused stack, not the linear GEMMs, and cannot be reconstructed by
replacing `nn.Linear` modules — these models require true FP8 hardware. See
[`scripts/sweep_fp8.py`](scripts/sweep_fp8.py) and the
[FP8-ROCM paper §5.6](https://github.com/wabibito/FP8-ROCM/blob/main/docs/PAPER.md).

## Hardware status

Brought up and validated on a Ryzen AI Max+ 395 (Radeon 8060S, gfx1151) with
ROCm 7.2.4 and the gfx1151 PyTorch wheel (torch 2.11.0+rocm7.13): the 1B and 7B
load and run on the iGPU, the flash-attn vortex patches are sufficient, and the
FP8 results above match the H100 reference. Remaining items to watch:

1. Memory behaviour for the 20B under ROCm's unified-memory model.
2. MIOpen emits a benign missing-tuning-DB warning on first FFT prefill; it falls
   back correctly.
3. The `gfx1151` wheel index in `install.sh` tracks AMD nightlies and may need a
   version bump over time.

**Follow [`docs/TESTING_PLAN.md`](docs/TESTING_PLAN.md)** — a phase-by-phase
bring-up plan with the exact commands, expected outputs, reference numbers, and
a known-risk register. It is written so that, on a fresh checkout on the 395,
you (or an assistant) can run a phase, paste the output, and continue debugging
from a known state.

## Credits

Evo 2 is by the Arc Institute. The FP8-on-non-FP8-hardware approach and the
test/verification scripts come from the Evo2MPS port. The general FP8-on-PyTorch
library is [FP8-MPS](https://github.com/wabibito/FP8-MPS).
