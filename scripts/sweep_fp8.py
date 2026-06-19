#!/usr/bin/env python3
"""
Sweep FP8 e4m3 emulation variants for evo2_1b_base and measure next-token
loss/accuracy on the bundled prompts vs the H100 reference.

The 1B is FP8-trained; the bf16 fallback collapses (~33%) and the default
emulation recovers most of it. This script isolates *which* knobs matter:
  - scale precision (bf16 vs fp32)
  - whether activations are quantized (full FP8 vs weight-only)
  - activation scale source (checkpoint delayed-scaling vs dynamic per-forward amax)
  - which layers are emulated (all FP8 layers vs input projections only)

    python scripts/sweep_fp8.py [--max-len 2048]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from importlib import resources

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H100 = {"loss": 0.501953125, "acc": 79.556}
FP8_MAX = 448.0


def quant_e4m3(x):
    from evo2.fp8_emulation import quantize_e4m3
    return quantize_e4m3(x)


class CfgFp8Linear(nn.Module):
    """Configurable e4m3-emulated linear for the sweep."""
    def __init__(self, weight, bias, act_scale, weight_scale, cfg, return_tuple, te_return_bias):
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))
        self.register_buffer("weight_scale", torch.tensor(float(weight_scale)))
        self.cfg = cfg
        self.return_tuple = return_tuple
        self.te_return_bias = te_return_bias

    def forward(self, x):
        w = self.weight
        cfg = self.cfg
        if cfg["scale_dtype"] == "fp32":
            xs, ws = x.float(), w.float()
        else:
            xs, ws = x.to(w.dtype), w
        # activation scale
        if cfg["act_mode"] == "dynamic":
            amax = xs.detach().abs().max().clamp_min(1e-6)
            act_scale = (FP8_MAX / amax).to(xs.dtype)
        elif cfg["act_mode"] == "clip_protect":
            # keep the trained (checkpoint) scale, but cap it so the current
            # tensor's max doesn't saturate e4m3 (448). Only lowers the scale
            # for prompts whose activations are larger than calibration assumed.
            amax = xs.detach().abs().max().clamp_min(1e-6)
            safe = (FP8_MAX / amax).to(xs.dtype)
            act_scale = torch.minimum(self.act_scale.to(xs.dtype), safe)
        else:
            act_scale = self.act_scale.to(xs.dtype)
        w_q = quant_e4m3(ws * self.weight_scale)
        accum = cfg.get("accum", "operand")  # "operand" = matmul in operand dtype; "fp32" = accumulate in fp32
        def lin(a, b):
            if accum == "fp32":
                return F.linear(a.float(), b.float())
            return F.linear(a, b)
        if cfg["quant_act"]:
            x_q = quant_e4m3(xs * act_scale)
            out = lin(x_q, w_q) * (1.0 / (act_scale * self.weight_scale))
        else:  # weight-only FP8
            out = lin(xs, w_q) * (1.0 / self.weight_scale)
        if self.bias is not None:
            out = out + self.bias
        out = out.to(w.dtype)
        if not self.return_tuple:
            return out
        return (out, self.bias) if self.te_return_bias else (out, None)


def apply_cfg(model, ckpt, cfg):
    from evo2.fp8_emulation import extract_fp8_scales
    scales = extract_fp8_scales(ckpt)
    modules = dict(model.named_modules())
    n = 0
    for path, sc in scales.items():
        if not cfg["include_dense"] and "mixer.dense" in path:
            continue
        parent_path, _, attr = path.rpartition(".")
        parent = modules.get(parent_path)
        if parent is None:
            continue
        old = getattr(parent, attr, None)
        if old is None or getattr(old, "weight", None) is None:
            continue
        is_te = hasattr(old, "te_return_bias")
        new = CfgFp8Linear(
            old.weight.data,
            old.bias.data if getattr(old, "bias", None) is not None else None,
            sc["act"], sc["weight"], cfg,
            return_tuple=is_te, te_return_bias=getattr(old, "te_return_bias", False),
        ).to(old.weight.device)
        setattr(parent, attr, new)
        n += 1
    return n


def read_prompts():
    with resources.path("evo2.test.data", "prompts.csv") as p:
        import csv
        seqs = []
        with open(p, encoding="utf-8-sig", newline="") as f:
            r = csv.reader(f); next(r)
            for row in r:
                if row:
                    seqs.append(row[0].strip())
    return seqs


def evaluate(model, seqs, max_len):
    rows = []
    for seq in seqs:
        if max_len:
            seq = seq[:max_len]
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(model.device)
        with torch.no_grad():
            fwd = model.model.forward(ids)
        logits = (fwd[0] if isinstance(fwd, (tuple, list)) else fwd)[0, :-1].float()
        tgt = ids[0, 1:].long()
        rows.append((F.cross_entropy(logits, tgt).item(),
                     (logits.argmax(-1) == tgt).float().mean().item() * 100))
    return rows


# Every "more precise / smarter" variant below was measured to HURT the
# FP8-trained 1B vs the plain default — the checkpoint depends on the exact
# trained e4m3 GEMM (bf16-operand, checkpoint delayed-scaling, saturation and
# all). Kept here as a reproducible record of what does NOT help.
CONFIGS = [
    ("bf16 fallback (no FP8)",            None),
    ("default (ckpt act-scale)",          dict(scale_dtype="bf16", quant_act=True,  act_mode="ckpt",         include_dense=True, accum="operand")),
    ("weight-only FP8 (act in bf16)",     dict(scale_dtype="bf16", quant_act=False, act_mode="ckpt",         include_dense=True, accum="operand")),
    ("dynamic act-scale (448/amax)",      dict(scale_dtype="bf16", quant_act=True,  act_mode="dynamic",      include_dense=True, accum="operand")),
    ("clip-protected act-scale",          dict(scale_dtype="bf16", quant_act=True,  act_mode="clip_protect", include_dense=True, accum="operand")),
    ("fp32 scaling",                      dict(scale_dtype="fp32", quant_act=True,  act_mode="ckpt",         include_dense=True, accum="operand")),
    ("bf16 grid + fp32-accum GEMM",       dict(scale_dtype="bf16", quant_act=True,  act_mode="ckpt",         include_dense=True, accum="fp32")),
]


def find_ckpt(model_name="evo2_1b_base"):
    import glob
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/**/{model_name}.pt"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(1)

    ckpt = find_ckpt()
    seqs = read_prompts()
    names = ["L1RE2", "ECOLAC", "Mammoth", "HumanMito"][:len(seqs)]
    os.environ["EVO2_FP8_EMULATION"] = "0"
    from evo2 import Evo2

    print(f"H100 ref (aggregate): loss={H100['loss']:.4f}  acc={H100['acc']:.3f}%   ({len(seqs)} prompts)\n")
    results = []
    for name, cfg in CONFIGS:
        m = Evo2("evo2_1b_base")
        n = 0 if cfg is None else apply_cfg(m.model, ckpt, cfg)
        rows = evaluate(m, seqs, args.max_len)
        loss = float(np.mean([r[0] for r in rows]))
        acc = float(np.mean([r[1] for r in rows]))
        results.append((name, loss, acc, n))
        print(f"== {name}  ({n} layers) ==")
        for nm, (l, a) in zip(names, rows):
            print(f"   {nm:<12} acc={a:6.2f}%  loss={l:.4f}")
        print(f"   {'AGGREGATE':<12} acc={acc:6.2f}%  loss={loss:.4f}   "
              f"(Δacc vs H100 {acc-H100['acc']:+.2f}pp)\n")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(results, key=lambda r: r[2])
    print("best:", best[0], f"({best[2]:.2f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
