# Evo2StrixHalo — Testing & Bring-Up Plan

This is the hand-off plan for validating Evo2StrixHalo on real AMD Strix Halo
hardware (Ryzen AI Max+ 395, `gfx1151`). The port was written
correct-by-construction but **has never run on a 395**. Work through the phases
below in order; each phase says what to run, what success looks like, and what
to paste back so the next session can fix any failure precisely.

**How to resume with an assistant:** check out this repo on the 395, run a
phase, and paste the *full* terminal output (including the command and any
traceback). Reference numbers come from the validated Evo2MPS port — they are
the targets to match within tolerance.

---

## Phase 0 — Environment sanity (no model download)

```bash
# ROCm sees the GPU?
rocminfo | grep -i "gfx\|Name:" | head
# If gfx1151 is not listed:
export HSA_OVERRIDE_GFX_VERSION=11.5.1

./install.sh
source .venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)              # should be non-None on ROCm
print("cuda.is_available:", torch.cuda.is_available())   # True on ROCm
print("device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
# the port's own device logic
from evo2.models import _is_rocm, _has_fp8_hardware, _get_default_device
print("_is_rocm:", _is_rocm())                # expect True
print("_has_fp8_hardware:", _has_fp8_hardware())  # expect False on RDNA 3.5
print("default device:", _get_default_device())   # expect cuda:0
PY
```

**Success:** `hip` is non-None, `cuda.is_available()` is True, `_is_rocm()` True,
`_has_fp8_hardware()` False, default device `cuda:0`.

**Likely failures & fixes:**
- `rocminfo` doesn't list the GPU → set `HSA_OVERRIDE_GFX_VERSION=11.5.1`.
- `install.sh` torch wheel index 404 / wrong ROCm version → override
  `EVO2_TORCH_INDEX` (see README) or point at the stable ROCm wheel index.
- `import vortex` fails → `vtx` may need a ROCm-compatible build; **report the
  full error** — this is the most likely first blocker.

**Paste back:** the whole Phase-0 output.

---

## Phase 1 — Patch application

```bash
python patches/patch_vortex.py            # auto-detects ROCm
```

**Success:** prints `backend: rocm` and applies the flash-attn fallback patches
(`optional_flash_attn`, `rotary_torch_fallback`, `qkv_view_path`) without
"pattern not found" errors.

**Likely failures & fixes:**
- "pattern for X not found" → the installed `vtx` version differs from what the
  patches target; **paste the output** and the `vtx` version
  (`pip show vtx`) so the patch patterns can be updated.
- If a later forward pass fails on a CUDA-API call (e.g. autocast, empty_cache),
  re-run `python patches/patch_vortex.py --backend mps` to apply the broader
  reroute set, then re-test.

**Paste back:** the patch output and `pip show vtx | grep -i version`.

---

## Phase 2 — Smoke test (1B, ~4 GB download)

```bash
python scripts/smoke_test.py --model evo2_1b_base
```

**Success:** loads, runs one forward pass on `cuda:0`, prints logits shape
`(1, 32, 512)` and `OK`. A warning that e4m3 emulation was applied to the input
projection(s) is expected for the 1B.

**Likely failures & fixes:**
- CUDA-API error mid-forward → apply the `--backend mps` patch set (Phase 1).
- HIP kernel / unsupported-op error → **paste the traceback**; may need a vortex
  op routed to a torch fallback (same pattern as the MPS patches).
- OOM → unlikely for the 1B; if so, note total system RAM.

**Paste back:** full output including any warning/traceback.

---

## Phase 3 — Full DNA pipeline (7B, the accurate workhorse, ~14 GB)

```bash
python scripts/test_dna.py --model evo2_7b_base
```

**Success:** all 6 stages print `OK` and "All checks passed." The 7B is
bf16-native, so no emulation and this is the real accuracy path.

**Paste back:** the 6-stage output.

---

## Phase 4 — Accuracy vs the H100 reference

```bash
python scripts/compare_to_upstream.py --model evo2_7b_base
```

**Target (must match within tolerance — same as Evo2MPS achieved):**

| Model | loss | acc | tolerance |
|-------|------|-----|-----------|
| evo2_7b_base | 0.3521 | 85.921% | loss ±0.05, acc ±1.5 pp |
| evo2_7b      | 0.3477 | 86.346% | same |

**Success:** "port matches upstream within tolerance." If the 7B matches H100
on ROCm the way it does on MPS, the core port is correct.

**Likely failures & fixes:**
- Large accuracy gap on the 7B → a numerical/device bug in the forward path
  (not FP8 — the 7B has no FP8). Run `scripts/verify_all.py` and **paste the
  table**; compare per-stage against MPS results.
- Memory pressure at 8K context → add `--max-len 2048` and note it.

**Paste back:** the comparison block.

---

## Phase 5 — FP8 emulation on the 1B

```bash
python scripts/validate_fp8_emulation.py --model evo2_1b_base
```

**Target:** bf16 fallback ~32% accuracy, e4m3 emulated **~74.5%** (within ~5 pp
of the 79.556% H100 reference). The emulation is device-agnostic, so it should
reproduce the MPS result on ROCm.

**Success:** "emulation meaningfully improves accuracy" and emulated acc in the
low-to-mid 70s.

**Paste back:** the aggregate block.

---

## Phase 6 — Full feature verification

```bash
python scripts/verify_all.py --full
```

**Success:** "ALL CHECKS PASS" — forward, scoring, variant/VEP, embeddings,
gene completion, and FP8 emulation all validated against reference data, exactly
as on the Mac.

**Paste back:** the full results table.

---

## Phase 7 — Optional: large models & web UI

```bash
# 20B/40B are NOT expected to be accurate (no FP8 hardware) but should load
# memory permitting. Confirm load + structured ACGT output only:
python scripts/smoke_test.py --model evo2_20b      # if RAM allows

# Web UI:
python webapp.py        # http://localhost:7860
```

The 20B near-random output is *expected* and documented; do not treat it as a
bug. The point of running it is to confirm load behaviour and memory limits on
Strix Halo's unified memory.

---

## Known risk register (most-likely-first)

1. **`vtx`/`vortex` ROCm build** — the runtime may have CUDA-only kernels.
   Highest-risk unknown. If `import vortex` or a forward pass fails on a HIP/op
   error, this is it.
2. **PyTorch gfx1151 wheel** — index/version drift; `install.sh` has a fallback
   but may need the current ROCm wheel URL.
3. **Vortex patch coverage** — ROCm may need more or fewer patches than the
   flash-attn subset; `--backend mps` is the broader fallback.
4. **Flash-attn absence** — configs already set `use_flash_attn: False`; if any
   path still imports flash-attn, the `optional_flash_attn` patch must catch it.
5. **Memory** — 20B (~40 GB) / 40B (~80 GB) depend on the box's unified-memory
   allocation; the 7B (~14 GB) and 1B (~4 GB) should be comfortable.

## Definition of done

- Phases 0–6 pass.
- `compare_to_upstream.py --model evo2_7b_base` matches H100 within tolerance
  (the 7B is the accuracy proof).
- `validate_fp8_emulation.py` recovers the 1B to the low-mid 70s.
- `verify_all.py --full` reports ALL CHECKS PASS.

At that point Evo2StrixHalo is a validated port and the README's "unverified"
banner can be removed.
