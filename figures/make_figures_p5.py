"""Figures for paper 5 (the speculative drafter).

Reads the archived JSONs in results/ (drafter_*.json) and writes into
papers/drafter/figs/. Same conventions as the other make_figures scripts:
every number in the paper regenerates from committed results.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "papers", "drafter", "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, GRAY, DARK = "#2b6ca3", "#9aa5ad", "#123c5f"


def load(name):
    return json.load(open(os.path.join(RES, name)))


# ---------------------------------------------------------------- fig 1 -----
# End-to-end T4 throughput, plain vs speculative, with acceptance labels.

configs = [
    ("drafter_gpu_A.json", "0.6B\nself"),
    ("drafter_gpu_C_17b.json", "1.7B\n+ state"),
    ("drafter_gpu_C_4b.json", "4B\n+ state"),
    ("drafter_gpu_B.json", "1.7B\nvanilla\n(control)"),
    ("drafter_gpu_gpt2.json", "GPT-2\nself"),
]
plain, spec, acc, speed = [], [], [], []
for f, _ in configs:
    r = load(f)["seen"]
    plain.append(r["plain"]["tok_per_s"])
    sp = r["spec:sillage"]
    spec.append(sp["tok_per_s"])
    acc.append(sp["accepted"] / max(1, sp["drafted"]))
    speed.append(sp["speedup"])

fig, ax = plt.subplots(figsize=(7.0, 3.4))
x = range(len(configs))
w = 0.38
ax.bar([i - w / 2 for i in x], plain, w, color=GRAY, label="greedy decoding")
ax.bar([i + w / 2 for i in x], spec, w, color=BLUE,
       label="speculative (identical output)")
for i in x:
    ax.text(i + w / 2, spec[i] + 1.2,
            f"×{speed[i]:.2f}\nacc {acc[i]:.0%}",
            ha="center", va="bottom", fontsize=8, color=DARK)
ax.set_xticks(list(x))
ax.set_xticklabels([lab for _, lab in configs], fontsize=9)
ax.set_ylabel("tokens / s  (T4, fp16, seen stream)")
ax.set_ylim(0, max(spec) * 1.35)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p5fig1_speedups.pdf"))
fig.savefig(os.path.join(OUT, "p5fig1_speedups.png"), dpi=160)
print("p5fig1_speedups: plain", [round(p, 1) for p in plain],
      "spec", [round(s, 1) for s in spec])

# ---------------------------------------------------------------- fig 2 -----
# The conversion factor: latency of a k-token forward with a warm cache.

lat = load("drafter_micro_t4.json")["latency_ms"]
ks = sorted(int(k) for k in lat)
ms = [lat[str(k)] for k in ks]

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.bar([str(k) for k in ks], ms, color=BLUE, width=0.6)
ax.axhline(ms[0], color=DARK, lw=0.8, ls="--")
for i, k in enumerate(ks):
    ax.text(i, ms[i] + 0.6, f"{ms[i] / ms[0]:.2f}×", ha="center",
            fontsize=8, color=DARK)
ax.set_xlabel("tokens verified in one forward (warm KV cache)")
ax.set_ylabel("latency (ms) — Qwen3-1.7B, T4, fp16")
ax.set_ylim(0, max(ms) * 1.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p5fig2_micro.pdf"))
fig.savefig(os.path.join(OUT, "p5fig2_micro.png"), dpi=160)
print("p5fig2_micro:", {k: round(v, 2) for k, v in zip(ks, ms)})

# ---------------------------------------------------------------- fig 3 -----
# Read-only calibration for a bigger target (dev NLL, 1.7B).

c = load("drafter_calib_17b.json")
labels = ["1.7B alone", "0.6B settings\n(inherited)",
          "calibrated for 1.7B\n(read-only, 2 min)"]
vals = [c["nll_base"], c["nll_old"], c["nll_new"]]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
bars = ax.bar(labels, vals, color=[GRAY, "#6f9dbf", BLUE], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}",
            ha="center", fontsize=9, color=DARK)
ax.set_ylabel("dev NLL (nats) — teacher-forced, doc 1")
ax.set_ylim(0, max(vals) * 1.22)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p5fig3_calibration.pdf"))
fig.savefig(os.path.join(OUT, "p5fig3_calibration.png"), dpi=160)
print("p5fig3_calibration:", [round(v, 4) for v in vals])
print("OK ->", OUT)
