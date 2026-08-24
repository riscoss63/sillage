"""Real end-to-end tests for the Sillage assistant (gpt2 backend for speed).

Each command runs in a SEPARATE subprocess = a real session, so persistence
is tested for real. The recall test uses invented facts the base model
cannot know. Run: python test_assistant.py  (~5 min on CPU).
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

import hashlib
import os
import re
import shutil
import subprocess
import sys

STATE = "test_state"
DOCS = "test_docs"
PY = sys.executable

FACTS = [
    "The Zylkorb protocol requires seventeen turquoise llamas.",
    "Project Marmelune deploys the vortex condenser at dawn.",
    "Captain Ilvress stores the amber cipher inside the ninth lighthouse.",
]
SUBJ = ["committee", "board", "council", "task force", "working group",
        "delegation", "panel", "office", "team", "department"]
VERB = ["reviewed", "discussed", "examined", "postponed", "approved",
        "rejected", "audited", "drafted", "archived", "circulated"]
OBJ = ["the quarterly report", "the budget allocation", "the hiring plan",
       "the maintenance schedule", "the safety audit", "the travel policy",
       "the vendor contract", "the training curriculum",
       "the archive migration", "the annual forecast"]


def filler_block(seed, sentences=90):
    """Varied filler, ~1300 GPT-2 tokens, so that repeated facts are farther
    apart than the model's 1024-token window: only persistent memory (not
    in-context copying) can predict their recurrences."""
    out = []
    for k in range(sentences):
        i = (seed * 31 + k * 7) % 10
        j = (seed * 17 + k * 13) % 10
        l = (seed * 23 + k * 3) % 10
        out.append(f"The {SUBJ[i]} {VERB[j]} {OBJ[l]} on day {k + 1} of "
                   f"session {seed + 1}.")
    return " ".join(out)


def make_docs():
    os.makedirs(DOCS, exist_ok=True)
    for name, off in [("doc_a.txt", 0), ("doc_a2.txt", 50)]:
        parts = []
        for rep in range(4):
            parts.append(filler_block(off + rep))
            for f in FACTS:
                parts.append(f)
        open(os.path.join(DOCS, name), "w", encoding="utf-8").write(
            "\n\n".join(parts))


def run(*args):
    cmd = [PY, "assistant.py", *args, "--model", "gpt2", "--state", STATE]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1200)
    assert r.returncode == 0, f"FAILED {args}: {r.stderr[-2000:]}"
    return r.stdout


def ppls(out):
    m = re.search(r"PPL frozen ([\d.]+) -> with memory ([\d.]+)", out)
    assert m, f"no PPL line in: {out}"
    return float(m.group(1)), float(m.group(2))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    shutil.rmtree(STATE, ignore_errors=True)
    make_docs()
    passed = []

    # T1 -- first read: runs, creates state
    out = run("read", os.path.join(DOCS, "doc_a.txt"))
    b1, m1 = ppls(out)
    assert os.path.exists(os.path.join(STATE, "state.npz"))
    assert os.path.exists(os.path.join(STATE, "cold.pkl"))
    passed.append(f"T1 first read ok (PPL {b1:.2f} -> {m1:.2f}); state saved")

    # T2 -- persistence across sessions
    out = run("status")
    m = re.search(r"lifetime tokens read : (\d+)", out)
    assert m and int(m.group(1)) > 500, out
    passed.append(f"T2 persistence ok ({m.group(1)} tokens remembered)")

    # T3 -- re-reading the same doc: memory must slash perplexity
    out = run("read", os.path.join(DOCS, "doc_a.txt"))
    b3, m3 = ppls(out)
    assert m3 < 0.8 * b3, f"expected big gain, got {b3} -> {m3}"
    passed.append(f"T3 same-doc memory gain ok ({b3:.2f} -> {m3:.2f})")

    # T4 -- cross-document transfer (edited sibling document)
    out = run("read", os.path.join(DOCS, "doc_a2.txt"))
    b4, m4 = ppls(out)
    assert m4 < 0.9 * b4, f"expected transfer gain, got {b4} -> {m4}"
    passed.append(f"T4 cross-doc transfer ok ({b4:.2f} -> {m4:.2f})")

    # T5 -- verbatim recall of an invented fact
    state_hash = sha(os.path.join(STATE, "state.npz"))
    out = run("complete", "The Zylkorb protocol requires", "-n", "8")
    assert "seventeen" in out, f"recall failed: {out}"
    passed.append("T5 invented-fact recall ok ('seventeen turquoise llamas')")

    # T6 -- generation must not modify the memory
    assert sha(os.path.join(STATE, "state.npz")) == state_hash
    passed.append("T6 no self-learning ok (state unchanged by generation)")

    # T7 -- forget wipes; completion still runs on the frozen model
    run("forget", "--all")
    assert not os.path.exists(os.path.join(STATE, "state.npz"))
    out = run("complete", "The Zylkorb protocol requires", "-n", "8")
    passed.append("T7 forget + base-only completion ok")

    print("\n".join(passed))
    print(f"\nALL {len(passed)} TESTS PASSED")


if __name__ == "__main__":
    main()
