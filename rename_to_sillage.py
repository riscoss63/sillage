"""One-shot project rename: SGAM -> Sillage (idempotent, verifiable).

Why: "SGAM" collides with the Smart Grid Architecture Model (IEC SRD 63200),
so the old acronym was unsearchable. The project is now Sillage -- French for
the trace left behind by something that has passed (a ship's wake, a scent in
a room), which is precisely what the memory keeps of what the model read.

Renames, in order:
  1. module files with historical names -> descriptive ones
  2. every import / `import X as Y` reference to them
  3. result files sgam_* -> sillage_* and the two loaders that read them
  4. prose and LaTeX: SGAM -> Sillage, \\sgam -> \\sillage, bibtex keys

Run once:  python rename_to_sillage.py
"""


# --- repo bootstrap ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_os.chdir(_d)
# --- end bootstrap ---

import glob
import io
import os
import re

MODULE_RENAMES = {
    "memory/sgam_semantic.py": "memory/sillage_semantic.py",
    "memory/sgam_router.py": "memory/sillage_router.py",
    "memory/bhd_v2_tune.py": "memory/key_selection.py",
    "memory/bhd_v2_final.py": "memory/ngram_memory.py",
    "memory/bhd_v3_distributional.py": "memory/sillage_factorial.py",
}
MODULE_TOKENS = {
    "sgam_semantic": "sillage_semantic",
    "sgam_router": "sillage_router",
    "bhd_v2_tune": "key_selection",
    "bhd_v2_final": "ngram_memory",
    "bhd_v3_distributional": "sillage_factorial",
}
PROSE = [
    (r"\\newcommand\{\\sgam\}\{\\textsc\{sgam\}\}",
     r"\\newcommand{\\sillage}{\\textsc{Sillage}}"),
    (r"\\sgam\b", r"\\sillage"),
    (r"\bsghairi2026sgam\b", "sghairi2026sillage"),
    (r"\bSGAM\b", "Sillage"),
    (r"\bsgam\b(?!\.tex|/)", "sillage"),
]
TEXT_EXT = (".py", ".md", ".tex", ".cff", ".txt")
SKIP = {"rename_to_sillage.py"}


def walk_files():
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "__pycache__", "data", "dumps",
                                "papers_state", "memory_state", "test_state",
                                "test_docs", "results"}]
        for fn in files:
            if fn.endswith(TEXT_EXT) and fn not in SKIP:
                yield os.path.join(root, fn)


def main():
    # 1. module files
    for src, dst in MODULE_RENAMES.items():
        if os.path.exists(src):
            os.rename(src, dst)
            print(f"renamed {src} -> {dst}")

    # 2-4. textual pass over code, docs and papers
    changed = 0
    for path in walk_files():
        s = io.open(path, encoding="utf-8").read()
        orig = s
        for old, new in MODULE_TOKENS.items():
            s = re.sub(r"\b" + old + r"\b", new, s)
        for pat, rep in PROSE:
            s = re.sub(pat, rep, s)
        if s != orig:
            io.open(path, "w", encoding="utf-8").write(s)
            changed += 1
            print(f"  patched {path}")

    # 3. result files + their loaders
    moved = 0
    for src in glob.glob("results/sgam_*.json"):
        dst = src.replace("sgam_", "sillage_", 1) if "results\\sgam_" in src \
            or "results/sgam_" in src else src
        dst = os.path.join("results",
                           os.path.basename(src).replace("sgam_", "sillage_", 1))
        if src != dst and not os.path.exists(dst):
            os.rename(src, dst)
            moved += 1
    print(f"renamed {moved} result files, patched {changed} text files")


if __name__ == "__main__":
    main()
