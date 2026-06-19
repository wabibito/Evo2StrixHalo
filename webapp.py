#!/usr/bin/env python3
"""
Gradio web UI for Evo 2 on Strix Halo.

Two tabs:
  - Generate: continue a DNA prompt (autoregressive sampling).
  - Score:    per-base log-likelihood of one or more sequences.

Models are loaded lazily and cached one-at-a-time (they are large; switching
models frees the previous one). On Strix Halo the iGPU is selected automatically
(ROCm reports through the CUDA API).

    conda activate Evo2StrixHalo   # or: source .venv/bin/activate
    python webapp.py               # then open http://127.0.0.1:7860

Notes:
  - evo2_1b_base is FP8-trained and loads with e4m3 emulation (fast, ~79% of the
    H100 next-token accuracy). evo2_7b_base is bf16-native and the accurate
    workhorse (matches the H100 reference) but slower. The 20B/40B are not
    numerically usable without FP8 hardware and are intentionally not offered.
"""

from __future__ import annotations

import os
import time

# Help the gfx1151 iGPU be detected before torch is imported (no-op elsewhere).
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

import gradio as gr

MODELS = ["evo2_1b_base", "evo2_7b_base"]
_CACHE: dict = {}


def get_model(name: str):
    """Load (and cache) one Evo2 model at a time."""
    if name not in _CACHE:
        from evo2 import Evo2
        _CACHE.clear()  # keep only one model resident
        _CACHE[name] = Evo2(name)
    return _CACHE[name]


def clean_dna(s: str) -> str:
    return "".join(c for c in s.upper() if c in "ACGTN")


def do_generate(model_name, prompt, n_tokens, temperature, top_k, top_p):
    prompt = clean_dna(prompt)
    if not prompt:
        return "", "Enter a DNA prompt using A/C/G/T."
    try:
        m = get_model(model_name)
        t0 = time.time()
        out = m.generate(
            [prompt], n_tokens=int(n_tokens), temperature=float(temperature),
            top_k=int(top_k), top_p=float(top_p), verbose=0,
        )
        # vortex returns a GenerationOutput (with .sequences); older paths a tuple.
        if hasattr(out, "sequences"):
            gen = out.sequences[0]
        elif isinstance(out, (tuple, list)):
            gen = out[0][0]
        else:
            gen = str(out)
        dt = time.time() - t0
        info = f"{model_name} on {m.device}: generated {len(gen)} bases in {dt:.1f}s"
        return prompt + gen, info
    except Exception as e:  # surface errors in the UI rather than the console only
        return "", f"Error: {type(e).__name__}: {e}"


def do_score(model_name, text):
    seqs = [clean_dna(line) for line in text.splitlines()]
    seqs = [s for s in seqs if s]
    if not seqs:
        return "Enter one or more DNA sequences (A/C/G/T), one per line."
    try:
        m = get_model(model_name)
        t0 = time.time()
        scores = m.score_sequences(seqs)
        dt = time.time() - t0
        lines = [f"{i+1}. len={len(s):>6}  score={sc:+.4f}"
                 for i, (s, sc) in enumerate(zip(seqs, scores))]
        lines.append(f"\n{model_name} on {m.device}: scored {len(seqs)} sequence(s) in {dt:.1f}s")
        lines.append("(score = mean per-token log-likelihood; higher = more plausible to the model)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


EXAMPLE = ("ATGGCGTGGCAACTGCTGCCGCCGCTGCTGCTGCTGCTGCTGGCGGGCGGCGGCGGC"
           "GGCTGGAGCGAACTGCTGGCGCTGCTGGCGCTGCTGCTGGCGCTGCTGGCG")


def build_demo():
    with gr.Blocks(title="Evo 2 — Strix Halo") as demo:
        gr.Markdown(
            "# Evo 2 on Strix Halo\n"
            "Genomic foundation model running on the AMD Radeon 8060S iGPU (RDNA 3.5). "
            "First request for a model triggers a download + load (slow); later requests reuse it."
        )
        model_dd = gr.Dropdown(MODELS, value=MODELS[0], label="Model")

        with gr.Tab("Generate"):
            gp = gr.Textbox(label="DNA prompt (A/C/G/T)", value=EXAMPLE, lines=3)
            with gr.Row():
                n_tokens = gr.Slider(16, 1024, value=128, step=16, label="tokens to generate")
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="temperature")
                top_k = gr.Slider(1, 16, value=4, step=1, label="top-k")
                top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top-p")
            gen_btn = gr.Button("Generate", variant="primary")
            gen_out = gr.Textbox(label="prompt + generated sequence", lines=6, show_copy_button=True)
            gen_info = gr.Markdown()
            gen_btn.click(do_generate, [model_dd, gp, n_tokens, temperature, top_k, top_p],
                          [gen_out, gen_info])

        with gr.Tab("Score"):
            sp = gr.Textbox(label="DNA sequence(s), one per line", value=EXAMPLE, lines=6)
            score_btn = gr.Button("Score", variant="primary")
            score_out = gr.Textbox(label="log-likelihood scores", lines=8, show_copy_button=True)
            score_btn.click(do_score, [model_dd, sp], [score_out])

    return demo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    build_demo().launch(server_name=args.host, server_port=args.port, share=args.share)
