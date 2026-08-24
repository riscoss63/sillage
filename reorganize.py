"""One-shot repository reorganization (idempotent).

Moves the flat script pile into role folders and prepends a small bootstrap
header to every moved file so that:
  * local imports keep working unchanged (`from memories import ...`)
  * every script can be launched from anywhere (it chdirs to the repo root,
    so data/, dumps/, results/ and ../paperN/figs all resolve as before)

Run once:  python reorganize.py
"""

import os
import shutil

LAYOUT = {
    "pipeline": ["data_prep.py", "data_prep_500k.py", "data_prep_qwen.py",
                 "dump_base.py", "dump_any.py", "dump_500k.py"],
    "memory": ["memories.py", "key_selection.py", "ngram_memory.py",
               "sillage_factorial.py", "sillage_semantic.py",
               "sillage_router.py", "multiseed.py", "multiseed_router.py",
               "exp_500k.py", "exp_500k_bigD.py", "router_500k.py",
               "hierarchy_500k.py", "continual.py", "continual_v2.py",
               "model2_qwen.py"],
    "fastweights": ["fastweights.py", "fastweights_combo.py",
                    "fastweights_scale.py"],
    "eval": ["cloze_eval.py", "cloze_router.py", "rag_baseline.py",
             "paired_test.py", "paired_test_v3.py", "diagnostic.py",
             "semantic_diag.py", "hier_diag.py", "smoke_test.py",
             "qwen_beta_ext.py"],
    "figures": ["make_figures.py", "make_figures_p2.py", "make_figures_p3.py",
                "make_figures_p4.py"],
}
ROOT_TOOLS = ["assistant.py", "demo.py", "test_assistant.py",
              "papers_assistant.py"]
MARKER = "# --- repo bootstrap"
BOOTSTRAP = '''{marker} (added by reorganize.py) ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \\
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
for _sub in ("", "pipeline", "memory", "fastweights", "eval", "figures"):
    _p = _os.path.join(_d, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)
# --- end bootstrap ---
'''.format(marker=MARKER)


def add_bootstrap(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        return False
    # insert after the module docstring if there is one
    idx = 0
    stripped = src.lstrip()
    if stripped.startswith('"""'):
        start = src.index('"""')
        end = src.index('"""', start + 3) + 3
        idx = end
        while idx < len(src) and src[idx] in "\r\n":
            idx += 1
    out = src[:idx] + ("\n" if idx else "") + BOOTSTRAP + "\n" + src[idx:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return True


def main():
    moved, patched = 0, 0
    for folder, files in LAYOUT.items():
        os.makedirs(folder, exist_ok=True)
        for fn in files:
            dst = os.path.join(folder, fn)
            if os.path.exists(fn):
                shutil.move(fn, dst)
                moved += 1
            if os.path.exists(dst) and add_bootstrap(dst):
                patched += 1
    for fn in ROOT_TOOLS:
        if os.path.exists(fn) and add_bootstrap(fn):
            patched += 1
    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__", ignore_errors=True)
    print(f"moved {moved} files, patched {patched} with the bootstrap header")


if __name__ == "__main__":
    main()
