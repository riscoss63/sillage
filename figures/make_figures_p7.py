"""Figures for paper 7 (LongMemEval). Reads results/lme_*.json, writes
papers/benchmark/figs/. Same convention as the other papers: every
number regenerates from committed JSON."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "papers", "benchmark", "figs")
os.makedirs(OUT, exist_ok=True)
BLUE, GRAY, DARK, RED = "#2b6ca3", "#9aa5ad", "#123c5f", "#a3452b"

SHORT = {"single-session-user": "user fact",
         "single-session-assistant": "assistant fact",
         "knowledge-update": "knowledge update",
         "multi-session": "multi-session",
         "temporal-reasoning": "temporal",
         "single-session-preference": "preference"}
ORDER = ["single-session-user", "single-session-assistant",
         "knowledge-update", "multi-session", "temporal-reasoning",
         "single-session-preference"]


def load(n):
    return json.load(open(os.path.join(RES, n)))


# fig 1 -- found vs formulated, per question type (arm E, 500 q) ----------
e = load("lme_arm_e.json")["metrics"]
ev = [e["by_type"][t]["evidence@3"] * 100 for t in ORDER]
an = [e["by_type"][t]["answer_top3"] * 100 for t in ORDER]
x = np.arange(len(ORDER))
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.bar(x - 0.19, ev, 0.38, color=BLUE, label="evidence session in top-3")
ax.bar(x + 0.19, an, 0.38, color=GRAY,
       label="answer string in top-3 (strict)")
for xi, (a, b) in enumerate(zip(ev, an)):
    ax.annotate(f"{a:.0f}", (xi - 0.19, a + 2), ha="center", fontsize=8,
                color=DARK)
    ax.annotate(f"{b:.0f}", (xi + 0.19, b + 2), ha="center", fontsize=8,
                color=DARK)
ax.axhline(e["overall"]["evidence@3"] * 100, color=BLUE, lw=0.8,
           ls="--", alpha=0.6)
ax.axhline(e["overall"]["answer_top3"] * 100, color=GRAY, lw=0.8,
           ls="--", alpha=0.9)
ax.annotate("63-point gap:\nfound $\\neq$ formulated",
            xy=(5.45, 60), ha="right", fontsize=9, color=RED)
ax.annotate("", xy=(5.55, e["overall"]["answer_top3"] * 100 + 1),
            xytext=(5.55, e["overall"]["evidence@3"] * 100 - 1),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
ax.set_xticks(x)
ax.set_xticklabels([SHORT[t] for t in ORDER], fontsize=8)
ax.set_ylabel("% of questions (470, judge-free)")
ax.set_ylim(0, 112)
ax.legend(frameon=False, fontsize=8, loc="upper right",
          bbox_to_anchor=(1.0, 1.02))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p7fig1_gap.pdf"))
fig.savefig(os.path.join(OUT, "p7fig1_gap.png"), dpi=160)
print("fig1:", [f"{v:.0f}/{w:.0f}" for v, w in zip(ev, an)])

# fig 2 -- the three voices (arm G, 43 q) ---------------------------------
g = load("lme_arm_g.json")
rows = [r for r in g["rows"] if not r["abs"]]
by = {t: [r for r in rows if r["type"] == t] for t in ORDER}
va = [100 * sum(r["a_mem"] for r in by[t]) / len(by[t]) for t in ORDER]
vb = [100 * sum(r["b_ctx_mem"] for r in by[t]) / len(by[t])
      for t in ORDER]
vc = [100 * sum(r["c_ctx_only"] for r in by[t]) / len(by[t])
      for t in ORDER]
x = np.arange(len(ORDER))
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.bar(x - 0.27, va, 0.26, color=DARK, label="(a) memory alone")
ax.bar(x, vb, 0.26, color=BLUE, label="(b) context + memory")
ax.bar(x + 0.27, vc, 0.26, color=GRAY, label="(c) context alone")
agree = sum(r["b_ctx_mem"] == r["c_ctx_only"] for r in rows)
ax.annotate(f"(b) $\\equiv$ (c) on {agree}/{len(rows)} questions:\n"
            "the memory is exactly neutral\nwhen the evidence is "
            "in the window",
            xy=(2.5, 82), ha="center", fontsize=9, color=RED)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[t] for t in ORDER], fontsize=8)
ax.set_ylabel("answer string in completion (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p7fig2_voices.pdf"))
fig.savefig(os.path.join(OUT, "p7fig2_voices.png"), dpi=160)
print("fig2:", list(zip(va, vb, vc)), "agree", agree)

# fig 3 -- ingestion throughput (the engineering) -------------------------
ir = load("lme_ingest_rates.json")["rates_tok_per_s"]
names = list(ir)
vals = [ir[n] for n in names]
labels = ["read_text\n(full scoring)", "per-token writes\n(fast_ingest)",
          "blocked GEMM\n(64-token blocks)"]
fig, ax = plt.subplots(figsize=(5.0, 3.2))
bars = ax.bar(range(3), vals, 0.55, color=[GRAY, GRAY, BLUE])
for i, v in enumerate(vals):
    ax.annotate(f"{v:.0f} tok/s", (i, v * 1.15), ha="center", fontsize=9,
                color=DARK)
ax.annotate(f"$\\times${vals[2]/vals[0]:.0f}", xy=(2, vals[2] * 2.4),
            ha="center", fontsize=13, color=RED)
ax.set_yscale("log")
ax.set_ylim(4, 3000)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("ingestion (tokens/s, log)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p7fig3_ingest.pdf"))
fig.savefig(os.path.join(OUT, "p7fig3_ingest.png"), dpi=160)
print("fig3:", vals, f"x{vals[2]/vals[0]:.0f}")
