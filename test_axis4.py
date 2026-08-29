"""The axis-4 commands, end to end: review, export, pull, watch.

Each one exists because a paper measured something, so each test checks
the measured behaviour rather than just the plumbing:

  review  paper 6's two-occurrence rule -- a document read twice must
          come out visibly more consolidated than one read once
  export  a cartridge must carry NO plain text, must still answer within
          the measured cost, and must refuse to pretend when the state is
          too thin to speak
  pull    the same cartridge, opened by somebody else: same answers, and
          three refusals -- never overwrite a memory, never open one
          written for another model, never unpickle a stranger's file
  watch   incremental reads with the SHIPPED extension filter, a salience
          journal that ranks a note full of new facts above routine prose,
          and two same-named notes in two subfolders that do not evict
          each other
  cli     the same commands through `python -m sillage`, because argparse
          wiring, exit codes and the printed block are what a user meets

The last two sections also guard two defects an audit found in 1.8.0: a
`watch` that read every file in a folder because the CLI shadowed its own
default, and a `complete --fast` that lost paper 8's recall on a --sem2
state while still claiming identical output.

    python test_axis4.py
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "behav"))

from sillage import Sillage                                    # noqa: E402
from sillage.index import strip_latex                          # noqa: E402
from sillage.watch import Watcher, watch                       # noqa: E402
from behavioral import (A_PREFIX, ENTS, VALS, build_doc)       # noqa: E402

PY = sys.executable
passed = []


def check(name, cond, detail=""):
    assert cond, f"{name} FAILED {detail}"
    passed.append(f"{name} ok {detail}")


def at(*parts):
    """Every scratch path is anchored on the file, not on the cwd."""
    return os.path.join(HERE, *parts)


def clean(*paths):
    for p in paths:
        p = at(p)
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (
            os.path.exists(p) and os.remove(p))


def write(rel, text, mode="w"):
    path = at(rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = {} if "b" in mode else {"encoding": "utf-8"}
    with open(path, mode, **kw) as f:
        f.write(text)
    return path


def run(*args):
    """One command line, in its own process, from the repository root."""
    p = subprocess.run([PY, "-m", "sillage", *args], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


SCRATCH = ("_a4_state", "_a4_out", "_a4_load", "_a4_pull", "_a4_pull2",
           "_a4_pull3", "_a4_cli", "_a4_notes", "_a4_dt", "_a4_sub",
           "_a4_once.md", "_a4_twice.md", "_a4_real.md", "_a4_facts.md",
           "_a4_small.md")

try:
    # --- A. review: the two-occurrence rule, visible -----------------------
    clean(*SCRATCH)
    write("_a4_once.md",
          build_doc(list(zip(ENTS[:6], VALS[:6])), seed=1, reps=1, block=40))
    write("_a4_twice.md",
          build_doc(list(zip(ENTS[6:12], VALS[6:12])), seed=2, reps=1,
                    block=40))
    s = Sillage(model="gpt2", state=at("_a4_state"), quiet=True)
    s.read(at("_a4_once.md"), fast=True)
    s.read(at("_a4_twice.md"), fast=True)
    s.read(at("_a4_twice.md"), fast=True)
    rows = {r["source"]: r for r in s.review()}
    once, twice = rows["_a4_once.md"], rows["_a4_twice.md"]
    check("A1 review sees the two-occurrence rule",
          twice["share"] > once["share"] + 0.2 and once["fragile"] > 0
          and twice["fragile"] == 0,
          f"(read once: {once['share']:.0%} consolidated, "
          f"{once['fragile']} fragile; read twice: {twice['share']:.0%}, "
          f"{twice['fragile']} fragile)")
    check("A2 review orders by what needs rereading",
          s.review()[0]["source"] == "_a4_once.md",
          "(least consolidated first)")
    clean("_a4_state", "_a4_once.md", "_a4_twice.md")

    # --- B. export and pull: the cartridge round trip ----------------------
    real = strip_latex(open(at("papers", "behavior", "behavior.tex"),
                            encoding="utf-8").read())[:12000]
    write("_a4_real.md", real)
    facts = list(zip(ENTS[:10], VALS[:10]))
    write("_a4_facts.md", build_doc(facts, seed=3, reps=3, block=40))
    s = Sillage(model="gpt2", state=at("_a4_state"), quiet=True,
                fastweights=False, sem2=5, sem2_whiten=True)
    s.read(at("_a4_real.md"), fast=True)
    s.read(at("_a4_facts.md"), fast=True)
    full = sum(v.split()[0] in s.complete(A_PREFIX.format(e=e), n=8)
               for e, v in facts)

    # paper 5's guarantee, on the tier paper 8 added: the fast path must
    # pool the same query at the same position, or it silently drops the
    # one impulse that makes paraphrased recall work
    probe = A_PREFIX.format(e=facts[0][0])
    check("B1 complete --fast is identical on a --sem2 state",
          s.complete(probe, n=8) == s.complete(probe, n=8, fast=True),
          "(paper 8's pooled impulse survives speculative decoding)")

    info = s.export_shareable(at("_a4_out"))
    blob = b"".join(open(at("_a4_out", f), "rb").read()
                    for f in info["files"])
    leaks = [w for w in (b"turquoise", b"Vorlagune", b"protocol requires")
             if w in blob]
    check("B2 no plain text in a cartridge", not leaks,
          f"({info['bytes']/1e6:.1f} MB, files: "
          f"{', '.join(info['files'])})")
    check("B3 the cold store and index are left out",
          "cold.npz" not in info["files"]
          and "index.json" not in info["files"]
          and info["manifest"]["left_out"],
          "(the two parts that hold text verbatim)")
    shutil.copytree(at("_a4_out"), at("_a4_load"), dirs_exist_ok=True)
    s2 = Sillage(model="gpt2", state=at("_a4_load"), quiet=True)
    cart = sum(v.split()[0] in s2.complete(A_PREFIX.format(e=e), n=8)
               for e, v in facts)
    # the published cost of dropping the cold store is two canonical
    # recalls in ten, so anything worse is a regression, not a tolerance
    check("B4 a cartridge still answers, within the measured cost",
          cart >= full - 2 and not info["thin"],
          f"(full state {full}/10, cartridge alone {cart}/10, "
          f"{len(s2.mem.cold)} cold grams and {len(s2.index.passages)} "
          f"passages shipped)")

    # a decoy in the source: pull must copy by name, never a directory
    write("_a4_out/notes.txt", "not part of a cartridge")
    s3 = Sillage(model="gpt2", state=at("_a4_pull"), quiet=True)
    got = s3.pull_cartridge(at("_a4_out"))
    pulled = sum(v.split()[0] in s3.complete(A_PREFIX.format(e=e), n=8)
                 for e, v in facts)
    check("B5 a pulled cartridge answers like the one exported",
          pulled == cart and got["manifest"]["hub"] == s.mem.hub,
          f"(pulled {pulled}/10 against {cart}/10 read from the same "
          f"files, written by sillage {got['manifest']['sillage']})")
    check("B6 pull copies the cartridge and nothing else",
          sorted(os.listdir(at("_a4_pull")))
          == sorted(Sillage.CARTRIDGE_FILES),
          "(a decoy file sitting beside it in the source stayed there)")
    try:
        Sillage(model="gpt2", state=at("_a4_pull"),
                quiet=True).pull_cartridge(at("_a4_out"))
        refused = False
    except RuntimeError as e:
        refused = "already holds a memory" in str(e)
    check("B7 pull never silently replaces a memory", refused,
          "(--force is required, and --state DIR keeps both)")
    try:
        Sillage(model="qwen", state=at("_a4_pull3"),
                quiet=True).pull_cartridge(at("_a4_out"))
        refused = False
    except RuntimeError as e:
        refused = "token space" in str(e)
    check("B8 pull refuses a cartridge from another model",
          refused and not os.path.isdir(at("_a4_pull3")),
          "(and says so before writing anything)")
    write("_a4_out/cold.pkl", b"not really a pickle", mode="wb")
    try:
        Sillage(model="gpt2", state=at("_a4_pull2"),
                quiet=True).pull_cartridge(at("_a4_out"))
        refused = False
    except RuntimeError as e:
        refused = "pre-1.5 pickle" in str(e)
    check("B9 pull refuses a stranger's pickle", refused,
          "(our own states migrate with a warning; a downloaded one is "
          "not ours to unpickle)")
    clean("_a4_state", "_a4_out", "_a4_load", "_a4_pull", "_a4_pull2",
          "_a4_pull3", "_a4_real.md", "_a4_facts.md")

    # a state too thin to speak must say so rather than ship a silent file
    write("_a4_small.md",
          build_doc(list(zip(ENTS[:4], VALS[:4])), seed=7, reps=1,
                    block=10))
    s = Sillage(model="gpt2", state=at("_a4_state"), quiet=True)
    s.read(at("_a4_small.md"), fast=True)
    info = s.export_shareable(at("_a4_out"))
    check("B10 export warns when the cartridge would be silent",
          info["thin"],
          f"({info['scored']} scored positions, under the 500 a tier "
          f"needs before it stops abstaining)")
    clean("_a4_state", "_a4_out", "_a4_small.md")

    # --- C. watch: incremental, selective, and the salience journal --------
    write("_a4_notes/routine.md",
          "The committee reviewed the quarterly report. " * 40)
    write("_a4_notes/sub/new-facts.md",
          build_doc(list(zip(ENTS[:6], VALS[:6])), seed=9, reps=1,
                    block=20))
    write("_a4_notes/.obsidian/workspace.json", "{}")
    write("_a4_notes/pic.png", b"\x89PNG not text", mode="wb")
    s = Sillage(model="gpt2", state=at("_a4_state"), quiet=True)
    # through watch(), with exts=None: exactly what the CLI passes when
    # --ext is absent, which is where the default used to be shadowed
    w = watch(s, at("_a4_notes"), once=True, exts=None, quiet=True)
    made = w.journal
    names = sorted(e["file"].replace("\\", "/") for e in made)
    check("C1 watch reads only what it should", len(made) == 2
          and names == ["routine.md", "sub/new-facts.md"],
          f"(read {names}; skipped .obsidian, a binary, and a .json)")
    check("C2 nothing changed, nothing read", w.pass_once() == [],
          "(a second pass is free)")
    write("_a4_notes/routine.md",
          "\n\nThe Zylkorb protocol requires seventeen turquoise brackets.",
          mode="a")
    again = w.pass_once()
    check("C3 an edited file is reread", len(again) == 1
          and again[0]["reread"] is True,
          "(and rereading is what consolidates it -- paper 6)")
    top = w.digest(2)
    check("C4 the salience journal ranks the new above the routine",
          top[0]["file"].replace("\\", "/").endswith("new-facts.md"),
          f"({top[0]['salience']:.2f} nats for invented facts against "
          f"{[round(e['salience'], 2) for e in top[1:]]} for prose the "
          f"model finds ordinary)")
    check("C5 the walk survives a restart",
          os.path.exists(at("_a4_state", "watch.json"))
          and Watcher(Sillage(model="gpt2", state=at("_a4_state"),
                              quiet=True),
                      at("_a4_notes"), quiet=True).pass_once() == [],
          "(state remembered, so a fresh process reads nothing again)")
    # a vault has many notes.md; they must not evict each other
    write("_a4_notes/a/notes.md",
          build_doc([(ENTS[7], VALS[7])], seed=11, reps=2, block=20))
    write("_a4_notes/b/notes.md",
          build_doc([(ENTS[8], VALS[8])], seed=12, reps=2, block=20))
    w.pass_once()
    sources = {p["source"].replace("\\", "/") for p in s.index.passages}
    check("C6 two notes of the same name keep their own passages",
          {"a/notes.md", "b/notes.md"} <= sources,
          f"(indexed as {sorted(x for x in sources if 'notes.md' in x)}, "
          f"not one basename overwriting the other)")
    clean("_a4_notes", "_a4_state")

    # --- D. the same commands through the command line ---------------------
    write("_a4_sub/deep/report.md",
          build_doc(list(zip(ENTS[:6], VALS[:6])), seed=13, reps=1,
                    block=40))
    rc, out = run("index", at("_a4_sub", "deep", "report.md"),
                  "--model", "gpt2", "--state", at("_a4_state"))
    rc2, ask = run("ask", ENTS[0], "--model", "gpt2",
                   "--state", at("_a4_state"))
    check("D1 index and ask work from the command line",
          rc == 0 and rc2 == 0 and "passages" in out and ENTS[0] in ask,
          "(no model is loaded for either)")

    rc, out = run("read", at("_a4_sub", "deep", "report.md"), "--fast",
                  "--model", "gpt2", "--state", at("_a4_state"))
    check("D2 status names a fast ingest instead of printing PPL None",
          rc == 0 and "fast ingest" in run(
              "status", "--model", "gpt2",
              "--state", at("_a4_state"))[1],
          "(a fast read has no perplexity to report, and says so)")

    rc, out = run("review", "--read", "1", "--model", "gpt2",
                  "--state", at("_a4_state"))
    check("D3 review --read rereads by the path it was read from",
          rc == 0 and "reread report.md" in out
          and "cannot reread" not in out,
          "(the document lives in a subdirectory, not in the cwd)")

    rc, out = run("export", at("_a4_out"), "--model", "gpt2",
                  "--state", at("_a4_state"))
    rc2, out2 = run("pull", at("_a4_out"), "--model", "gpt2",
                    "--state", at("_a4_cli"))
    rc3, out3 = run("pull", at("_a4_out"), "--model", "gpt2",
                    "--state", at("_a4_cli"))
    check("D4 export and pull round-trip on the command line",
          rc == 0 and rc2 == 0 and "cartridge written" in out
          and "pulled" in out2 and rc3 == 1
          and "already holds a memory" in out3,
          "(and the second pull exits 1 rather than replacing a memory)")

    rc, out = run("complete", "the report", "-n", "4", "--model", "gpt2",
                  "--state", at("_a4_state"), "--dtype", "int8")
    rc2, out2 = run("complete", "the report", "-n", "4", "--model", "gpt2",
                    "--state", at("_a4_state"), "--dtype", "bfloat16")
    check("D5 --dtype loads, and says what it costs",
          rc == 0 and "Conv1D" in out and "int8" in out
          and rc2 == 0 and "4x slower" in out2,
          "(int8 says how much of GPT-2 it could not touch; bfloat16 "
          "loads and prints the measured caveat)")
    clean("_a4_state", "_a4_out", "_a4_cli", "_a4_sub")
finally:
    clean(*SCRATCH)
    clean(".sillage-demo")

print("\n".join(passed))
print(f"\nALL {len(passed)} AXIS-4 TESTS PASSED")
