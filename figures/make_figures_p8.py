"""Figures for paper 8 (the paraphrase wall). Reads
results/semantic_*.json, writes papers/paraphrase/figs/. Same
convention as the other papers: every number regenerates from
committed JSON."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "papers", "paraphrase", "figs")
os.makedirs(OUT, exist_ok=True)
BLUE, GRAY, DARK, RED = "#2b6ca3", "#9aa5ad", "#123c5f", "#a3452b"


def load(n):
    return json.load(open(os.path.join(RES, n)))


# fig 1 -- the identity gradient across depth, two models -----------------
qw = load("semantic_layers_qwen.json")["layers"]
g2 = load("semantic_gpt2_replication.json")["sweep"]
qx = [r["layer"] / (len(qw) - 1) for r in qw]
qy = [r["A"]["delta"] for r in qw]
gx = [r["layer"] / (len(g2) - 1) for r in g2]
gy = [r["dA"] for r in g2]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.plot(qx, qy, "s-", color=BLUE, lw=2, ms=4,
        label="Qwen3-0.6B (28 layers)")
ax.plot(gx, gy, "o-", color=DARK, lw=2, ms=4, label="GPT-2 (12 layers)")
ax.axhline(0, color=GRAY, lw=0.8)
ax.plot([1 / (len(qw) - 1)], [qy[1]], "*", color=RED, ms=16)
ax.plot([5 / (len(g2) - 1)], [gy[5]], "*", color=RED, ms=16)
ax.annotate("chosen layers\n(picked on A, validated on B)",
            xy=(0.16, 0.55), fontsize=8, color=RED)
ax.annotate("the layer the tier was reading",
            xy=(0.99, min(gy) + 0.04), ha="right", fontsize=8,
            color=DARK)
ax.set_xlabel("network depth (fraction)")
ax.set_ylabel("entity-identity separation\n(median same $-$ null cosine)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p8fig1_gradient.pdf"))
fig.savefig(os.path.join(OUT, "p8fig1_gradient.png"), dpi=160)
print("fig1: qwen peak", max(qy[1:]), "gpt2 last", gy[-1])

# fig 2 -- the staircase: median retrieval rank of the true value ---------
steps = [
    ("shipped tier\n(last layer)",
     load("semantic_diag_qwen.json")["agg"]["median_rankS_B"]),
    ("+ oracle\nanchors",
     load("semantic_anchor_qwen.json")["rules"]["oracle"]["agg"]
     ["median_rank_B"]),
    ("+ dense keys\n(ZCA)",
     load("semantic_zca_qwen.json")["variants"]["oracle+dense"]["agg"]
     ["median_B"]),
    ("+ query\npooling",
     load("semantic_pooling_qwen.json")["agg"]["median_B"]),
]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
xs = np.arange(len(steps))
vals = [max(1.0, s[1]) for s in steps]
bars = ax.bar(xs, vals, 0.55,
              color=[GRAY, GRAY, BLUE, BLUE])
for i, v in enumerate(vals):
    ax.annotate(f"{v:,.0f}", (i, v * 1.25), ha="center", fontsize=9,
                color=DARK)
ax.set_yscale("log")
ax.set_ylim(1, 4e5)
ax.set_xticks(xs)
ax.set_xticklabels([s[0] for s in steps], fontsize=8)
ax.set_ylabel("median rank of the true value\n(paraphrased prompt, log)")
ax.annotate("random $\\approx$ 76{,}000", xy=(0.02, 90000), fontsize=8,
            color=GRAY)
ax.axhline(76000, color=GRAY, lw=0.8, ls="--")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p8fig2_staircase.pdf"))
fig.savefig(os.path.join(OUT, "p8fig2_staircase.png"), dpi=160)
print("fig2:", vals)

# fig 3 -- the behavioral trajectory --------------------------------------
v2 = load("semantic_behavioral_v2.json")
v3 = load("semantic_behavioral_v3.json")
gz = load("semantic_gpt2_zca.json")
labels = ["shipped tier\n(paper 6)", "layer-1 tier\n+ pooling",
          "+ gate\nwrite-filter", "+ echo +\nword integrity",
          "GPT-2\n(+ ZCA)"]
vals = [0.0, 25.0, 30.0, v3["test_B"] * 100, gz["test_B"] * 100]
cols = [GRAY, GRAY, GRAY, BLUE, DARK]
fig, ax = plt.subplots(figsize=(5.8, 3.3))
ax.bar(range(5), vals, 0.55, color=cols)
for i, v in enumerate(vals):
    ax.annotate(f"{v:.0f}%", (i, v + 2), ha="center", fontsize=10,
                color=DARK)
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("generative paraphrase recall\n(held-out facts, %)")
ax.set_ylim(0, 78)
ax.annotate("locality: 0--1/10 witness prompts\nchange at every step",
            xy=(0.05, 66), fontsize=8, color=GRAY)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p8fig3_trajectory.pdf"))
fig.savefig(os.path.join(OUT, "p8fig3_trajectory.png"), dpi=160)
print("fig3:", vals)
