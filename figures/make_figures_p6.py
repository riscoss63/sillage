"""Figures for paper 6 (behavioral laws). Reads results/behav_*.json,
writes papers/behavior/figs/. Same convention as the other papers: every
number regenerates from committed JSON."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "papers", "behavior", "figs")
os.makedirs(OUT, exist_ok=True)
BLUE, GRAY, DARK, RED = "#2b6ca3", "#9aa5ad", "#123c5f", "#a3452b"


def load(n):
    return json.load(open(os.path.join(RES, n)))


# fig 1 -- conflict dynamics and the readout-trust probe -------------------
gq = load("behav_qwen.json")
curve = load("behav_qwen_curve.json")
probe = load("behav_qwen_readout_probe.json")
gg = load("behav_gpt2.json")

qx = [1, 2, 3, 4]
qy = [gq["conflict_after_1"]["new"], gq["conflict_after_2"]["new"],
      curve["x3"]["new"], curve["x4"]["new"]]
gx = [1, 2]
gy = [gg["conflict_after_1"]["new"], gg["conflict_after_2"]["new"]]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(gx, [v * 100 for v in gy], "o-", color=GRAY, lw=2,
        label="GPT-2, published readout")
ax.plot(qx, [v * 100 for v in qy], "s-", color=BLUE, lw=2,
        label="Qwen3-0.6B, published readout")
ax.plot([4], [probe["calibrated"]["new"] * 100], "*", color=RED,
        markersize=16, label="same state, calibrated-trust readout\n"
        "(no new reads)")
ax.annotate("", xy=(4, 96), xytext=(4, 26),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax.set_xlabel("readings of the corrected document (v2)")
ax.set_ylabel("new value recalled (%)")
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(-4, 108)
ax.legend(frameon=False, fontsize=8, loc="center left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p6fig1_trust.pdf"))
fig.savefig(os.path.join(OUT, "p6fig1_trust.png"), dpi=160)
print("fig1:", gy, qy, probe["calibrated"]["new"])

# fig 2 -- context equivalence, two regimes --------------------------------
rec = load("behav_equivalence_gpt2.json")["caps"]
tra = load("behav_equivalence_gpt2_transfer.json")
caps = sorted(int(c) for c in rec)

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharex=True)
for ax, data, title in (
        (axes[0], rec, "recitation (documents the state read)"),
        (axes[1], tra["caps"], "transfer (edited sibling, never read)")):
    ax.plot(caps, [data[str(c)]["ppl_bare"] for c in caps], "o-",
            color=GRAY, lw=2, label="bare model")
    ax.plot(caps, [data[str(c)]["ppl_mem"] for c in caps], "s-",
            color=BLUE, lw=2, label="model + state")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context cap C (tokens)")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("test perplexity")
axes[0].legend(frameon=False, fontsize=9)
axes[0].annotate("never caught:\nthe answer is not\nin the local window",
                 xy=(1024, 65), xytext=(90, 22), fontsize=8, color=DARK,
                 arrowprops=dict(arrowstyle="->", color=DARK, lw=1))
cstar = tra["c_star_bare"]
axes[1].axvline(cstar, color=RED, ls="--", lw=1.2)
axes[1].annotate(f"C* = {cstar:.0f}", xy=(cstar, 8), fontsize=9,
                 color=RED, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p6fig2_equivalence.pdf"))
fig.savefig(os.path.join(OUT, "p6fig2_equivalence.png"), dpi=160)
print("fig2: C* =", cstar)

# fig 3 -- who carries retention -------------------------------------------
cr = load("behav_cold_retention_gpt2.json")

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharey=True)
for ax, arm, title in ((axes[0], "nodecay", "no forgetting"),
                       (axes[1], "decay", "half-life 30k tokens")):
    pts = cr[arm]
    xs = [p["interference_tokens"] / 1000 for p in pts]
    for key, color, style, lab in (
            ("full", DARK, "-", "full system"),
            ("no_matrix", BLUE, "--", "cold store + adapter"),
            ("no_cold", GRAY, "-.", "matrix + adapter"),
            ("base_adapter", RED, ":", "adapter alone")):
        ax.plot(xs, [p[key] * 100 for p in pts], style, color=color,
                marker="o", lw=2, label=lab)
    ax.set_xlabel("interference (thousands of tokens)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-5, 108)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("fact recall (%)")
axes[1].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p6fig3_retention.pdf"))
fig.savefig(os.path.join(OUT, "p6fig3_retention.png"), dpi=160)
print("fig3 ok ->", OUT)
