"""Paper-3 figures (routed hierarchy + surprise-gated consolidation).

fig1  consolidation Pareto: dNLL vs cold-store entries (log x), three
      admission policies, with the no-cold control, the unbounded ceiling
      and the unbounded kNN-LM reference
fig2  three-tier architecture schematic (one surprise signal, three jobs)
"""


# --- repo bootstrap (added by reorganize.py) ---
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
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})
R = "results"
OUT = os.path.join("papers", "hierarchy", "figs")
os.makedirs(OUT, exist_ok=True)
FRACS = [0.001, 0.005, 0.02, 0.1, 0.3]
KNN_REF = {"bible": 0.0575, "tolstoy": 0.0475}


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- figure 1 --
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
for ax, (dom, title) in zip(axes, [("bible", "King James Bible"),
                                   ("tolstoy", "War and Peace")]):
    j = load(f"hierarchy_{dom}.json")
    for pol, c, ls in [("surprise", BLUE, "-"), ("count", ORANGE, "-"),
                       ("random", GRAY, "-")]:
        xs = [j[f"{pol}@{f}"]["cold_entries"] for f in FRACS]
        ys = [j[f"{pol}@{f}"]["dnll_test"] for f in FRACS]
        ax.plot(xs, ys, ls, color=c, lw=1.8, marker="o", ms=3.5, zorder=3)
        ax.annotate(pol, (xs[-1], ys[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=7.2, color=c,
                    va="center")
    ceil = j["cold_unbounded"]["dnll_test"]
    ctrl = j["no_cold"]["dnll_test"]
    ax.axhline(ceil, color=INK, lw=0.9, ls=":", zorder=2)
    ax.axhline(ctrl, color=GRAY, lw=0.9, ls="--", zorder=2)
    ax.axhline(KNN_REF[dom], color=AQUA, lw=1.2, ls="-.", zorder=2)
    x0 = j["surprise@0.001"]["cold_entries"] * 0.8
    ax.annotate(f"unbounded cold ceiling ({j['cold_unbounded']['cold_entries']//1000}k grams)",
                (x0, ceil), xytext=(0, 3), textcoords="offset points",
                fontsize=6.6, color=INK)
    ax.annotate("no cold tier (paper-2 router)", (x0, ctrl), xytext=(0, 3),
                textcoords="offset points", fontsize=6.6, color=GRAY)
    ax.annotate("unbounded kNN-LM, 770 MB", (x0, KNN_REF[dom]),
                xytext=(0, -9), textcoords="offset points", fontsize=6.6,
                color=AQUA)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("cold-store entries (log)")
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("test ΔNLL (nats)")
save(fig, "p3fig1_pareto")

# ---------------------------------------------------------------- figure 2 --
BLUE_BG, ORANGE_BG, AQUA_BG = "#e5effb", "#fdeee7", "#e2f5ee"


def box(ax, x, y, w, h, text, fc, ec, fs=8.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           lw=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3)


def arrow(ax, p, q, color=INK, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 color=color, ls=ls, lw=lw, zorder=4))


fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")
box(ax, 2, 40, 18, 8, "token stream", "#f4f4f1", "#c3c2b7")
box(ax, 2, 22, 18, 9, "frozen LM", "#f4f4f1", "#c3c2b7")
box(ax, 28, 39, 26, 9, "FAST $n$-gram tier $M_G$\n4.2 MB, one-shot Hebbian",
    BLUE_BG, BLUE)
box(ax, 28, 26, 26, 9, "FAST semantic tier $M_S$\nbanded SimHash, 12.6 MB",
    BLUE_BG, BLUE)
box(ax, 28, 6, 26, 11,
    "COLD store, bounded\nexact 4-grams admitted at\nsurprise-mass threshold",
    AQUA_BG, AQUA)
box(ax, 66, 20, 28, 12,
    "confidence router\n$\\sum_i [m_i]\\,\\lambda_i p_i + \\mathrm{rest}\\cdot p_{LM}$",
    "#f4f4f1", "#898781")
box(ax, 66, 42, 28, 7, "prediction $p(x_{j+1})$", "#f4f4f1", "#c3c2b7")
arrow(ax, (20, 44), (28, 43.5))
arrow(ax, (20, 42), (28, 30.5))
arrow(ax, (54, 43), (70, 32))
arrow(ax, (54, 30), (66, 27))
arrow(ax, (54, 11.5), (66, 22))
arrow(ax, (80, 32), (80, 42))
arrow(ax, (20, 24), (66, 24), color=GRAY, lw=1.0)
ax.text(58, 25.3, "$p_{LM}$", fontsize=7.4, color=INK2)
ax.add_patch(FancyArrowPatch((28, 41), (28, 13),
                             connectionstyle="arc3,rad=0.42",
                             arrowstyle="-|>", mutation_scale=11,
                             color=ORANGE, ls="--", lw=1.6, zorder=4))
ax.text(13, 35, "consolidate", fontsize=7.2, color=ORANGE, ha="center")
ax.text(11, 12, "surprise $g=-\\ln p_{LM}$ gates\nwrites, routing,\nand consolidation",
        fontsize=7.4, color=ORANGE, ha="center")
save(fig, "p3fig2_architecture")
