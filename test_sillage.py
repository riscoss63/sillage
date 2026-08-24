"""End-to-end tests for the Sillage assistant (GPT-2 backend, for speed).

Every command runs in a SEPARATE subprocess -- a real session -- so
persistence is tested for real, not simulated. Recall is tested with invented
facts the base model cannot know, placed farther apart than the model's
1024-token window, so only the memory (never in-context copying) can predict
their recurrences.

    python test_sillage.py       (~20 min on a laptop CPU; see
                                  test_unit.py for the 5-second version)
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "test_state")
DOCS = os.path.join(HERE, "test_docs")
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


def run(*args, state=STATE, expect=0):
    cmd = [PY, "-m", "sillage", *args, "--model", "gpt2", "--state", state]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE,
                       encoding="utf-8", errors="replace", timeout=2400)
    assert r.returncode == expect, f"FAILED {args}: {r.stderr[-2000:]}"
    return r.stdout


def ppls(out):
    """(frozen, adapter, +memory) from a `read` line."""
    m = re.search(r"PPL ([\d.]+)(?: -> ([\d.]+) \(adapter\))?"
                  r" -> ([\d.]+) \(\+memory\)", out)
    assert m, f"no PPL line in: {out}"
    fw = float(m.group(2)) if m.group(2) else None
    return float(m.group(1)), fw, float(m.group(3))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    shutil.rmtree(STATE, ignore_errors=True)
    make_docs()
    doc_a = os.path.join(DOCS, "doc_a.txt")
    doc_a2 = os.path.join(DOCS, "doc_a2.txt")
    passed = []

    # T1 -- first read: runs, creates state, all four mechanisms present
    out = run("read", doc_a)
    b1, f1, m1 = ppls(out)
    for f in ("state.npz", "cold.pkl", "index.pkl", "log.json"):
        assert os.path.exists(os.path.join(STATE, f)), f"missing {f}"
    with np.load(os.path.join(STATE, "state.npz")) as z:
        assert z["M"].shape == (4096, 256) and np.abs(z["M"]).sum() > 0
        assert z["A"].shape == (50257, 16) and np.abs(z["A"]).sum() > 0
        assert int(z["g_cnt"]) > 500
        m_abs_one_pass = float(np.abs(z["M"]).sum())   # same doc, no decay
    passed.append(f"T1 first read ok (PPL {b1:.2f} -> {f1:.2f} adapter "
                  f"-> {m1:.2f} memory); M_G, A, cold store and index saved")

    # T2 -- persistence across sessions
    out = run("status")
    n = int(re.search(r"read so far\s*:\s*(\d+)", out).group(1))
    assert n > 500, out
    assert "fast weights" in out and "cold store" in out
    passed.append(f"T2 persistence ok ({n} tokens remembered across "
                  f"processes)")

    # T3 -- re-reading the same document: memory must slash perplexity
    out = run("read", doc_a)
    b3, f3, m3 = ppls(out)
    assert m3 < 0.8 * b3, f"expected a big memory gain, got {b3} -> {m3}"
    passed.append(f"T3 same-doc memory gain ok ({b3:.2f} -> {m3:.2f})")

    # T4 -- the readout adapter alone must help, without any memory mixing
    assert f3 < b3, f"fast weights did not help: {b3} -> {f3}"
    passed.append(f"T4 fast-weights gain ok ({b3:.2f} -> {f3:.2f}, "
                  f"adapter only)")

    # T5 -- cross-document transfer (edited sibling document)
    out = run("read", doc_a2)
    b5, _, m5 = ppls(out)
    assert m5 < 0.9 * b5, f"expected transfer gain, got {b5} -> {m5}"
    passed.append(f"T5 cross-doc transfer ok ({b5:.2f} -> {m5:.2f})")

    # T6 -- verbatim recall of an invented fact the base model cannot know
    state_hash = sha(os.path.join(STATE, "state.npz"))
    out = run("complete", "The Zylkorb protocol requires", "-n", "8")
    assert "seventeen" in out, f"recall failed: {out}"
    passed.append("T6 invented-fact recall ok ('seventeen turquoise llamas')")

    # T7 -- generation must never modify the memory
    assert sha(os.path.join(STATE, "state.npz")) == state_hash
    passed.append("T7 no self-learning ok (state unchanged by generation)")

    # T8 -- grounded retrieval quotes the document, and only it
    out = run("ask", "amber cipher lighthouse")
    assert "Ilvress" in out and "doc_a" in out, out
    passed.append("T8 grounded ask ok (exact passage, with its source)")

    # T9 -- forgetting: decay shrinks the traces, --all wipes everything
    st2 = os.path.join(HERE, "test_state_decay")
    shutil.rmtree(st2, ignore_errors=True)
    run("read", doc_a, "--half-life", "500", state=st2)
    with np.load(os.path.join(st2, "state.npz")) as zd:
        assert float(zd["half_life"]) == 500.0
        assert float(np.abs(zd["M"]).sum()) < m_abs_one_pass, \
            "decay did not shrink the matrix"
    shutil.rmtree(st2, ignore_errors=True)
    passed.append("T9 forgetting ok (half-life persisted, traces decayed)")

    # T10 -- the Python API is the same object as the CLI
    api = os.path.join(HERE, "test_state_api")
    shutil.rmtree(api, ignore_errors=True)
    code = (
        "import json, sys; sys.path.insert(0, %r)\n"
        "from sillage import Sillage\n"
        "s = Sillage(model='gpt2', state=%r, quiet=True)\n"
        "r = s.read(%r)[0]\n"
        "hit = s.ask('turquoise llamas', k=1)\n"
        "print(json.dumps({'ppl': r['ppl_with_memory'], "
        "'src': hit[0][1]['source'] if hit else None, "
        "'tok': s.status()['tokens']}))\n"
        % (HERE, api, doc_a))
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       cwd=HERE, encoding="utf-8", errors="replace",
                       timeout=2400)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["src"] == "doc_a.txt" and got["tok"] > 500, got
    shutil.rmtree(api, ignore_errors=True)
    passed.append(f"T10 Python API ok (read + ask + status, "
                  f"PPL {got['ppl']})")

    # T11 -- forget wipes; completion still runs on the frozen model alone
    run("forget", "--all")
    assert not os.path.exists(os.path.join(STATE, "state.npz"))
    run("complete", "The Zylkorb protocol requires", "-n", "8")
    passed.append("T11 forget + base-only completion ok")

    print("\n".join(passed))
    print(f"\nALL {len(passed)} TESTS PASSED")


if __name__ == "__main__":
    main()
