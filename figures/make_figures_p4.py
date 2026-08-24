"""Paper-4 figures (fast weights complement memory).

fig1  regime split: memory vs fast weights vs their combination, on the
      memory-dominant streams and on the low-repetition ones
fig2  long-horizon stability: per-50k segment gains over a 500k stream
fig3  two ablations: adapter rank sweep, and the failed surprise gate
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

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
GRAY_L, INK, INK2, GRID = "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})
R = "results"
OUT = os.path.join("papers", "fastweights", "figs")
os.makedirs(OUT, exist_ok=True)
NL = chr(10)


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def fw_key(d):
    for k in ("fw_r256_uniform", "fw_eta0.1_uniform", "fw_r64", "fw_r16"):
        if k in d:
            return k
    raise KeyError(d.keys())


# ---------------------------------------------------------------- figure 1 --
MEM_DOM = [("GPT-2" + NL + "Manuscripts", "fwcombo_bhd_run1.json"),
           ("Qwen3" + NL + "Manuscripts", "fwscale_q_bhd.json")]
FW_DOM = [("GPT-2" + NL + "Einstein", "fwcombo_relativity.json"),
          ("GPT-2" + NL + "Alice", "fwcombo_alice.json"),
          ("GPT-2" + NL + "W&P 500k", "fwscale_tolstoy.json"),
          ("Qwen3" + NL + "Einstein", "fwscale_q_relativity.json")]


def trio_panel(ax, rows, ylim, title, legend=False, fs=6.8):
    w = 0.26
    for gi, (label, fn) in enumerate(rows):
        d = load(fn)
        trio = [("memory", d["memory_only"]["dnll_test"], AQUA),
                ("fast weights", d[fw_key(d)]["dnll_test"], ORANGE),
                ("both", d["fw_plus_memory"]["dnll_test"], BLUE)]
        for si, (nm, v, c) in enumerate(trio):
            xp = gi + (si - 1) * w
            ax.bar(xp, v, width=w * 0.9, color=c, zorder=3,
                   label=nm if (gi == 0 and legend) else None)
            ax.text(xp, v + ylim * 0.02, f"{v:.3f}", ha="center",
                    fontsize=fs, color=INK)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.8)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("test ΔNLL (nats)")
    ax.set_title(title, fontsize=8.6)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                         gridspec_kw={"width_ratios": [1, 1.75]})
trio_panel(axes[0], MEM_DOM, 0.80,
           "novel, highly repetitive: memory leads", legend=True, fs=7.2)
axes[0].legend(frameon=False, fontsize=7.4, loc="upper right", ncol=1)
trio_panel(axes[1], FW_DOM, 0.108, "low repetition: the ordering flips")
save(fig, "p4fig1_regimes")

# ---------------------------------------------------------------- figure 2 --
d = load("fwscale_tolstoy.json")
x = [25 + 50 * i for i in range(10)]
fig, ax = plt.subplots(figsize=(5.6, 2.6))
series = [("fast weights (3.2 MB)", d["fw_r16"]["segments"], ORANGE, 0),
          ("both", d["fw_plus_memory"]["segments"], BLUE, 9),
          ("memory (4.2 MB)", d["memory_only"]["segments"], AQUA, 0)]
for name, seg, c, dy in series:
    ax.plot(x, seg, "-o", color=c, lw=1.8, ms=3, zorder=3)
    ax.annotate(name, (x[-1], seg[-1]), xytext=(6, dy),
                textcoords="offset points", fontsize=7, color=c, va="center")
ax.axhline(0, color=GRAY_L, lw=0.8, zorder=1)
ax.set_xlabel("stream position (thousands of tokens)")
ax.set_ylabel("ΔNLL / 50k segment (nats)")
ax.set_xlim(0, 700)
ax.set_ylim(0, 0.095)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("500k tokens: no drift, no divergence", fontsize=8.6)
save(fig, "p4fig2_stability")

# ---------------------------------------------------------------- figure 3 --
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
al = load("fwcombo_alice.json")
qb = load("fwscale_q_bhd.json")
qr = load("fwscale_q_relativity.json")
rel = load("fwcombo_relativity.json")

ax = axes[0]
curves = [("GPT-2 · Alice", [16, 64, 256],
           [al["fw_r16_uniform"]["dnll_test"],
            al["fw_r64_uniform"]["dnll_test"],
            al["fw_r256_uniform"]["dnll_test"]], BLUE, 0),
          ("GPT-2 · Einstein", [64, 256],
           [rel["fw_r64_uniform"]["dnll_test"],
            rel["fw_r256_uniform"]["dnll_test"]], GRAY, 7),
          ("Qwen3 · Einstein", [16, 64],
           [qr["fw_r16"]["dnll_test"], qr["fw_r64"]["dnll_test"]], ORANGE, -7),
          ("Qwen3 · Manuscripts", [16, 64],
           [qb["fw_r16"]["dnll_test"], qb["fw_r64"]["dnll_test"]], AQUA, 0)]
for name, xs, ys, c, dy in curves:
    ax.plot(xs, ys, "-o", color=c, lw=1.6, ms=4, zorder=3, label=name)
ax.legend(frameon=False, fontsize=6.8, loc="lower right")
ax.set_xscale("log", base=2)
ax.set_xticks([16, 64, 256])
ax.set_xticklabels(["16", "64", "256"])
ax.set_xlim(13, 330)
ax.set_ylim(0, 0.10)
ax.set_xlabel("adapter rank r")
ax.set_ylabel("test ΔNLL (nats)")
ax.set_title("rank 16 is enough", fontsize=8.6)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
bh = load("fwcombo_bhd_run1.json")
pairs = [("Manuscripts", bh["fw_eta0.1_gated"]["dnll_test"],
          bh["fw_eta0.1_uniform"]["dnll_test"]),
         ("Einstein", rel["fw_r256_gated"]["dnll_test"],
          rel["fw_r256_uniform"]["dnll_test"])]
for gi, (nm, gated, unif) in enumerate(pairs):
    ax.bar(gi - 0.17, gated, width=0.32, color=GRAY, zorder=3,
           label="surprise-gated" if gi == 0 else None)
    ax.bar(gi + 0.17, unif, width=0.32, color=ORANGE, zorder=3,
           label="uniform, same budget" if gi == 0 else None)
    ax.text(gi - 0.17, max(gated, 0) + 0.008, f"{gated:+.3f}", ha="center",
            fontsize=7, color=INK)
    ax.text(gi + 0.17, unif + 0.008, f"{unif:+.3f}", ha="center",
            fontsize=7, color=INK)
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([p[0] for p in pairs], fontsize=8)
ax.set_ylim(-0.02, 0.33)
ax.axhline(0, color=GRAY_L, lw=0.8, zorder=1)
ax.set_ylabel("test ΔNLL (nats)")
ax.set_title("surprise gating hurts fast weights", fontsize=8.6)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7.2, loc="upper center")
save(fig, "p4fig3_ablations")
