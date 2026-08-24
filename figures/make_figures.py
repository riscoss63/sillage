"""Paper figures (PDF + PNG), generated from the results JSONs.

fig1  main results on the Manuscripts stream (horizontal bars + 95% CI)
fig2  500k scaling curves per 50k segment (tolstoy | bible panels)
fig3  mechanism attribution on Manuscripts (bars + 95% CI)

Palette: validated reference palette (dataviz skill) — categorical
blue/orange/aqua in fixed order; the Sillage family is coded by one hue with
linestyle variants; baselines in muted gray; direct labels everywhere.
"""


# --- repo bootstrap: run this script from anywhere ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
for _sub in ("", "pipeline", "memory", "fastweights", "eval", "figures"):
    _p = _os.path.join(_d, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)
# --- end bootstrap ---

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_L = "#86b6ef"
GRAY, INK, INK2, GRID = "#898781", "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})

R = "results"
OUT = os.path.join("papers", "sillage", "figs")
os.makedirs(OUT, exist_ok=True)


def load(name):
    return json.load(open(os.path.join(R, name)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- figure 1 --
v3 = load("bhd_v3_bhd.json")
mem = load("memories_bhd.json")
fin = load("bhd_v2_final_bhd.json")
rag = load("rag_bhd.json")

rows = [
    ("Sillage (amp values, system gate)", v3["amp_system_n4"], BLUE),
    ("Exact 4-gram dictionary", fin["ngram_dict"], GRAY),
    ("kNN-LM (unbounded, 55 MB)", mem["knn"], GRAY),
    ("Sillage counts (ablation)", v3["count_model_n4"], BLUE_L),
    ("kNN-LM (byte-matched, 4.2 MB)", fin["knn_cap_matched"], GRAY),
    ("Unigram cache (null)", mem["cache_unigram"], GRAY),
    ("Sillage uniform writes (ablation)", fin["bhd_v2_unif"], BLUE_L),
]
rows.sort(key=lambda r: r[1]["dnll_test"])
labels = [r[0] for r in rows]
vals = [r[1]["dnll_test"] for r in rows]
cis = [r[1]["dnll_ci95"] for r in rows]
cols = [r[2] for r in rows]
rag_v = rag["base_nll_test_same_positions"] - rag["rag_interp"]["nll_test"]
labels.insert(0, "RAG-lite (retrieve + rescore)")
vals.insert(0, rag_v)
cis.insert(0, None)
cols.insert(0, GRAY)

fig, ax = plt.subplots(figsize=(6.4, 3.1))
y = range(len(labels))
ax.barh(y, vals, height=0.62, color=cols, zorder=3)
for i, (v, ci) in enumerate(zip(vals, cis)):
    if ci:
        ax.plot(ci, [i, i], color=INK, lw=1.0, zorder=4)
    anchor = (ci[1] if ci else v) + 0.012
    ax.text(anchor, i, f"+{v:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Test NLL improvement over frozen GPT-2 (nats)")
ax.set_xlim(0, 0.74)
ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
save(fig, "fig1_main")

# ---------------------------------------------------------------- figure 2 --
t5 = load("exp500k_tolstoy.json")
b5 = load("exp500k_bible.json")
big = load("exp500k_bible_D16384.json")
x = [25 + 50 * i for i in range(10)]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=False)
panels = [("War and Peace (narrative)", t5, None),
          ("King James Bible (formulaic)", b5, big)]
for ax, (title, j, extra) in zip(axes, panels):
    series = [("kNN-LM (770 MB)", j["knn"]["segments_dnll"], BLUE, "-", 0),
              ("4-gram dict", j["ngram_dict"]["segments_dnll"], ORANGE,
               "-", -7 if extra is None else 0),
              ("Sillage 4.2 MB", j["bhd_amp"]["segments_dnll"], AQUA, "-",
               -2 if extra is None else 0),
              ("Sillage + decay", j["bhd_amp_decay"]["segments_dnll"],
               AQUA, "--", 6 if extra is None else 0)]
    if extra:
        series.append(("Sillage 16.8 MB", extra["segments_dnll"], AQUA, ":", 0))
    for name, seg, c, ls, dy in series:
        ax.plot(x, seg, color=c, ls=ls, lw=1.8, zorder=3)
        ax.annotate(name, (x[-1], seg[-1]), xytext=(4, dy),
                    textcoords="offset points", fontsize=6.8, color=c,
                    va="center")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("stream position (thousands of tokens)")
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 640)
axes[0].set_ylabel("ΔNLL vs frozen LM (nats / segment)")
save(fig, "fig2_scaling")

# ---------------------------------------------------------------- figure 3 --
rows3 = [
    ("uniform writes\n(no gating)", fin["bhd_v2_unif"], BLUE_L),
    ("counts,\nmodel gate", v3["count_model_n4"], BLUE_L),
    ("counts,\nsystem gate", v3["count_system_n4"], BLUE_L),
    ("amplitudes,\nmodel gate", v3["amp_model_n4"], BLUE_L),
    ("amplitudes,\nsystem gate", v3["amp_system_n4"], BLUE),
]
fig, ax = plt.subplots(figsize=(4.6, 2.6))
xs = range(len(rows3))
for i, (name, r, c) in enumerate(rows3):
    ax.bar(i, r["dnll_test"], width=0.6, color=c, zorder=3)
    ax.plot([i, i], r["dnll_ci95"], color=INK, lw=1.0, zorder=4)
    ax.text(i, r["dnll_ci95"][1] + 0.022, f"+{r['dnll_test']:.3f}",
            ha="center", fontsize=8, color=INK)
ax.set_xticks(list(xs))
ax.set_xticklabels([r[0] for r in rows3], fontsize=7.6)
ax.set_ylabel("ΔNLL (nats)")
ax.set_ylim(0, 0.72)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig3_mechanisms")

# ---------------------------------------------------------------- figure 4 --
GRAY_L = "#c3c2b7"
groups = [("GPT-2\nManuscripts", load("cloze_bhd.json")),
          ("Qwen3-0.6B\nManuscripts", load("cloze_q_bhd.json")),
          ("GPT-2\nTolstoy 500k", load("cloze_tolstoy.json"))]
systems = [("frozen LM", "base", GRAY), ("+ kNN-LM", "knn", GRAY_L),
           ("+ Sillage", "sillage", BLUE)]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
w = 0.26
for gi, (gname, j) in enumerate(groups):
    for si, (sname, key, c) in enumerate(systems):
        r = j["all"][key]
        xpos = gi + (si - 1) * w
        ax.bar(xpos, 100 * r["acc"], width=w * 0.92, color=c, zorder=3,
               label=sname if gi == 0 else None)
        lo, hi = [100 * v for v in r["wilson95"]]
        ax.plot([xpos, xpos], [lo, hi], color=INK, lw=1.0, zorder=4)
        ax.text(xpos, hi + 0.7, f"{100 * r['acc'] + 1e-3:.1f}", ha="center",
                fontsize=7.4, color=INK)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("content-token recall (top-1 %)")
ax.set_ylim(0, 30)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "fig4_downstream")

# ------------------------------------------------- figure 0: architecture --
from matplotlib.patches import FancyArrowPatch, Rectangle


def box(ax, x, y, w, h, text, fc, ec, fs=8.2, tc=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3)


def arrow(ax, p, q, color=INK, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 color=color, ls=ls, lw=lw, zorder=4))


BLUE_BG, ORANGE_BG = "#e5effb", "#fdeee7"
fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")

box(ax, 2, 30, 20, 9, "token stream\n$\\ldots\\,x_{j-3}\\,x_{j-2}\\,x_{j-1}\\,x_j$", "#f4f4f1", "#c3c2b7")
box(ax, 2, 5, 20, 9, "frozen LM\n(no gradients ever)", "#f4f4f1", "#c3c2b7")
box(ax, 27, 30, 26, 9, "sliding $n$-gram binding\n$k_j=\\rho(k_{j-1})\\odot T_{x_j}\\odot\\rho^n T_{x_{j-n}}$", BLUE_BG, BLUE)
box(ax, 60, 27, 12, 15, "$M$\n$D_K{\\times}D_V$\n4.2 MB fixed", BLUE_BG, BLUE)
box(ax, 78, 30, 20, 9, "readout\n$p_{\\mathrm{mem}}=\\sigma(\\beta\\, Vu/\\|u\\|)$", BLUE_BG, BLUE)
box(ax, 78, 5, 20, 9, "mix + abstain\n$\\lambda p_{\\mathrm{mem}}+(1{-}\\lambda)p_{\\mathrm{LM}}$", "#f4f4f1", "#898781")
box(ax, 30, 5, 22, 9, "surprise gate\n$g=-\\ln p_{\\mathrm{LM}}(x_{j+1})$", ORANGE_BG, ORANGE)

arrow(ax, (22, 34.5), (27, 34.5))
arrow(ax, (53, 34.5), (60, 34.5))
arrow(ax, (72, 34.5), (78, 34.5))
arrow(ax, (88, 30), (88, 14))
arrow(ax, (12, 30), (12, 14))
arrow(ax, (22, 9.5), (30, 9.5))
arrow(ax, (52, 9.5), (64, 25.8), color=ORANGE, ls="--")
ax.text(43, 19.5, "amplitude write\n$M \\leftarrow M+(\\sqrt{a^2+g}-a)\\,q\\otimes V_{x_{j+1}}$",
        fontsize=7.6, color=ORANGE, ha="center")
ax.text(24.5, 12.2, "$p_{\\mathrm{LM}}$", fontsize=8, color=INK2)
ax.text(89.5, 21, "$p_{\\mathrm{mem}}$", fontsize=8, color=INK2)
save(fig, "fig0_architecture")
