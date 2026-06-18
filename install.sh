#!/usr/bin/env bash
#
# Set up Evo2StrixHalo on AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S /
# gfx1151) with ROCm on Ubuntu.
#
# Baseline target: Ubuntu 26.04, ROCm 7.x, PyTorch ROCm build for gfx1151.
#
# What this does:
#   1. Checks for ROCm + a usable GPU (rocminfo).
#   2. Creates a Python venv.
#   3. Installs the gfx1151 PyTorch ROCm wheels.
#   4. Installs `vtx` (the StripedHyena 2 runtime, imported as `vortex`).
#   5. Installs this repo (`evo2` package) editable, without deps.
#   6. Applies patches/patch_vortex.py (ROCm subset: flash-attn fallbacks).
#
# NOTE: gfx1151 is not on AMD's official ROCm support matrix as of early 2026,
# but ROCm 7.x works in practice. If the GPU is not detected, set:
#   export HSA_OVERRIDE_GFX_VERSION=11.5.1
# Re-runnable: each step skips if already done.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${EVO2_VENV:-$REPO_ROOT/.venv}"
# gfx1151-specific PyTorch ROCm wheel index (AMD nightlies). Override if needed.
TORCH_INDEX="${EVO2_TORCH_INDEX:-https://rocm.nightlies.amd.com/v2/gfx1151/}"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[setup]\033[0m %s\n" "$*" >&2; }

# 1. ROCm / GPU check
if ! command -v rocminfo >/dev/null 2>&1; then
  err "rocminfo not found. Install ROCm first:"
  err "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
  err "Then re-run this script."
  exit 1
fi
if ! rocminfo 2>/dev/null | grep -qi "gfx1151\|gfx11"; then
  err "No gfx11xx GPU detected by rocminfo. If you have a Strix Halo, try:"
  err "  export HSA_OVERRIDE_GFX_VERSION=11.5.1"
  err "and re-run. Continuing anyway..."
fi
log "ROCm detected: $(rocminfo 2>/dev/null | grep -m1 -i 'Name:.*gfx' | xargs || echo 'gfx11xx')"

# 2. venv
if [[ ! -d "$VENV" ]]; then
  log "creating venv at $VENV ..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "python: $(python -V) at $(which python)"
pip install --upgrade pip >/dev/null

# 3. PyTorch (ROCm gfx1151)
if python -c "import torch; assert torch.version.hip" >/dev/null 2>&1; then
  log "torch (ROCm) already installed: $(python -c 'import torch; print(torch.__version__, torch.version.hip)')"
else
  log "installing PyTorch ROCm wheels for gfx1151 from $TORCH_INDEX ..."
  pip install --index-url "$TORCH_INDEX" torch torchvision torchaudio || {
    err "gfx1151 nightly index failed; falling back to the stable ROCm index."
    err "Performance may be suboptimal (built for gfx1100, runs on gfx1151)."
    pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
  }
fi

# 4. vtx (StripedHyena 2 runtime; imported as `vortex`)
if python -c "import vortex" >/dev/null 2>&1; then
  log "vortex already installed"
else
  log "installing vtx (StripedHyena 2 runtime)..."
  pip install vtx
fi

# 5. Editable install of the evo2 package (no deps; keep our patched source)
log "installing local Evo2StrixHalo package (editable, no deps)..."
pip install --no-deps -e "$REPO_ROOT"
pip install biopython huggingface_hub pyyaml "einops>=0.8" packaging rich tqdm numpy
pip install pandas openpyxl   # BRCA1 VEP / batch scoring
pip install "gradio>=4.0,<6"  # webapp.py

# 6. Apply runtime vortex patches (ROCm subset auto-detected)
log "applying vortex patches (ROCm flash-attn fallbacks)..."
python "$REPO_ROOT/patches/patch_vortex.py"

cat <<EOF

------------------------------------------------------------
Evo2StrixHalo setup complete.

Activate the env:
  source "$VENV/bin/activate"

If the GPU is not detected at runtime, export:
  export HSA_OVERRIDE_GFX_VERSION=11.5.1

Smoke test (downloads ~4 GB on first run):
  python scripts/smoke_test.py --model evo2_1b_base

Full DNA test:
  python scripts/test_dna.py --model evo2_7b_base

7B-8k checkpoints run in bf16 and are the accurate workhorse. The 1B is
FP8-trained; it loads with e4m3 emulation (Strix Halo has no FP8 hardware).
The 20B/40B are not numerically usable without FP8 hardware (see README).
------------------------------------------------------------
EOF
