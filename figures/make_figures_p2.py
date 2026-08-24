"""Paper-2 figures (mixture of memories / semantic keys).

fig1  diagnostics: (a) hidden-state cosine geometry raw vs whitened;
      (b) band-match precision curves vs the top-32 retrieval ceiling
fig2  G-only -> router dumbbells across every (model, domain) configuration
fig3  500k segment curves for the router (requires router500k_*.json;
      skipped gracefully if not yet present)

Same validated palette and conventions as paper 1.
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
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})
R = "results"
OUT = os.path.join("papers", "router", "figs")
os.makedirs(OUT, exist_ok=True)


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- figure 1 --
dA = load("semantic_diag_alice.json")
dT = load("semantic_diag_tolstoy.json")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

ax = axes[0]
rows = [
    ("random pairs, raw", dA["raw_centered"]["random_mean"],
     dA["raw_centered"]["random_p95"], GRAY),
    ("random pairs, whitened", dA["whitened"]["random_mean"],
     dA["whitened"]["random_p95"], BLUE),
    ("same-next-token, whitened", dA["whitened"]["same_next_mean"],
     None, AQUA),
]
for i, (name, mean, p95, c) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([mean], [y], "o", color=c, ms=7, zorder=4)
    if p95 is not None:
        ax.plot([mean, p95], [y, y], color=c, lw=2.2, zorder=3)
        ax.plot([p95], [y], "|", color=c, ms=11, mew=2.2, zorder=4)
        ax.annotate(f"p95 {p95:+.2f}", (p95, y), xytext=(4, 5),
                    textcoords="offset points", fontsize=7.2, color=c)
    dy = -11 if p95 is not None else 7
    ax.annotate(f"{mean:+.2f}", (mean, y), xytext=(-2, dy),
                textcoords="offset points", fontsize=7.2, color=c,
                ha="center")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8)
ax.set_xlabel("cosine similarity (Alice, GPT-2)")
ax.set_xlim(-0.1, 1.0)
ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
ax.set_title("whitening collapses the false-match tail", fontsize=8.6)

ax = axes[1]
for d, name, ls in [(dA, "Alice", "-"), (dT, "War and Peace", "--")]:
    ms = [1, 2, 4, 8]
    pr = [d["simhash_b16"]["precision_by_min_bands"][str(m)] for m in ms]
    ax.plot(ms, pr, ls, color=BLUE, lw=1.8, marker="o", ms=4, zorder=3)
    ax.annotate(name, (ms[-1], pr[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=7.4, color=BLUE,
                va="center")
    ax.axhline(d["top32_reference"]["precision"], color=GRAY, lw=1.1,
               ls=ls, zorder=2)
ax.annotate("top-32 retrieval ceiling", (1.05, dA["top32_reference"]
            ["precision"] + 0.015), fontsize=7.2, color=GRAY)
ax.set_xlabel("minimum matching bands m (b = 16)")
ax.set_ylabel("P(same next token)")
ax.set_xticks([1, 2, 4, 8])
ax.set_ylim(0, 0.72)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("band matches grade into a precision kernel", fontsize=8.6)
save(fig, "p2fig1_diagnostics")

# ---------------------------------------------------------------- figure 2 --
configs = [
    ("GPT-2 · Manuscripts", "multiseed_router_bhd.json", "ms"),
    ("GPT-2 · Tolstoy 100k", "multiseed_router_tolstoy.json", "ms"),
    ("GPT-2 · Einstein", "sillage_router_relativity_multi.json", "single"),
    ("GPT-2 · Alice", "sillage_router_alice_multi.json", "single"),
    ("Qwen3 · Manuscripts", "sillage_router_q_bhd_multi_nw.json", "single"),
    ("Qwen3 · Alice", "sillage_router_q_alice_multi_nw.json", "single"),
]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
labels = []
for i, (name, fn, kind) in enumerate(configs):
    j = load(fn)
    if kind == "ms":
        g, r = j["g_only"]["mean"], j["router"]["mean"]
    else:
        g, r = j["g_only"]["dnll_test"], j["router"]["dnll_test"]
    y = len(configs) - 1 - i
    ax.plot([g, r], [y, y], color="#c3c2b7", lw=2.0, zorder=2)
    ax.plot([g], [y], "o", color=GRAY, ms=6, zorder=3)
    ax.plot([r], [y], "o", color=BLUE, ms=7, zorder=4)
    ax.annotate(f"{r:+.3f}", (r, y), xytext=(6, -3),
                textcoords="offset points", fontsize=7.4, color=BLUE)
    labels.append(name)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(list(reversed(labels)), fontsize=8)
ax.set_xscale("symlog", linthresh=0.02)
ax.set_xlabel("test ΔNLL (nats, symlog scale) — gray: n-gram only, "
              "blue: + semantic routing")
ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
save(fig, "p2fig2_router_gains")

# ---------------------------------------------------------------- figure 3 --
try:
    panels = []
    for dom, title in [("tolstoy", "War and Peace"),
                       ("bible", "King James Bible")]:
        panels.append((title, load(f"router500k_{dom}.json"),
                       load(f"router500k_{dom}_decay.json")))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    x = [25 + 50 * i for i in range(10)]
    for pi, (ax, (title, nod, dec)) in enumerate(zip(axes, panels)):
        dy_g, dy_r = (0, 0) if pi == 0 else (-7, 6)
        series = [("n-gram only", nod["g_only"]["segments"], GRAY, "-", dy_g),
                  ("router", nod["router"]["segments"], BLUE, "-", dy_r),
                  ("router + decay", dec["router"]["segments"], BLUE,
                   "--", -7 if pi == 0 else 0)]
        for name, seg, c, ls, dy in series:
            ax.plot(x, seg, ls, color=c, lw=1.8, zorder=3)
            ax.annotate(name, (x[-1], seg[-1]), xytext=(4, dy),
                        textcoords="offset points", fontsize=6.8, color=c,
                        va="center")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("stream position (k tokens)")
        ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 660)
    axes[0].set_ylabel("ΔNLL / segment (nats)")
    save(fig, "p2fig3_scaling")
except FileNotFoundError as e:
    print("fig3 skipped (500k results pending):", e)
