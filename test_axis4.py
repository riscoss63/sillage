"""The axis-4 commands, end to end: review, export, watch.

Each one exists because a paper measured something, so each test checks
the measured behaviour rather than just the plumbing:

  review  paper 6's two-occurrence rule -- a document read twice must
          come out visibly more consolidated than one read once
  export  a cartridge must carry NO plain text, must still answer, and
          must refuse to pretend when the state is too thin to speak
  watch   incremental reads, and a salience journal that ranks a note
          full of new facts above routine prose

    python test_axis4.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "behav"))

from sillage import Sillage                                    # noqa: E402
from sillage.index import strip_latex                          # noqa: E402
from sillage.watch import Watcher                              # noqa: E402
from behavioral import (A_PREFIX, ENTS, VALS, build_doc)       # noqa: E402

passed = []


def check(name, cond, detail=""):
    assert cond, f"{name} FAILED {detail}"
    passed.append(f"{name} ok {detail}")


def clean(*paths):
    for p in paths:
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (
            os.path.exists(p) and os.remove(p))


# --- A. review: the two-occurrence rule, visible ---------------------------
clean("_a4_state", "_a4_once.md", "_a4_twice.md")
open("_a4_once.md", "w", encoding="utf-8").write(
    build_doc(list(zip(ENTS[:6], VALS[:6])), seed=1, reps=1, block=40))
open("_a4_twice.md", "w", encoding="utf-8").write(
    build_doc(list(zip(ENTS[6:12], VALS[6:12])), seed=2, reps=1,
              block=40))
s = Sillage(model="gpt2", state="_a4_state", quiet=True)
s.read("_a4_once.md", fast=True)
s.read("_a4_twice.md", fast=True)
s.read("_a4_twice.md", fast=True)
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

# --- B. export: a cartridge that carries no text and still answers ---------
clean("_a4_state", "_a4_out", "_a4_load", "_a4_real.md", "_a4_facts.md")
real = strip_latex(open(os.path.join("papers", "behavior",
                                     "behavior.tex"),
                        encoding="utf-8").read())[:12000]
open("_a4_real.md", "w", encoding="utf-8").write(real)
facts = list(zip(ENTS[:10], VALS[:10]))
open("_a4_facts.md", "w", encoding="utf-8").write(
    build_doc(facts, seed=3, reps=3, block=40))
s = Sillage(model="gpt2", state="_a4_state", quiet=True,
            fastweights=False, sem2=5, sem2_whiten=True)
s.read("_a4_real.md", fast=True)
s.read("_a4_facts.md", fast=True)
full = sum(v.split()[0] in s.complete(A_PREFIX.format(e=e), n=8)
           for e, v in facts)
info = s.export_shareable("_a4_out")
blob = b"".join(open(os.path.join("_a4_out", f), "rb").read()
                for f in info["files"])
leaks = [w for w in (b"turquoise", b"Vorlagune", b"protocol requires")
         if w in blob]
check("B1 no plain text in a cartridge", not leaks,
      f"({info['bytes']/1e6:.1f} MB, files: "
      f"{', '.join(info['files'])})")
check("B2 the cold store and index are left out",
      "cold.npz" not in info["files"]
      and "index.json" not in info["files"]
      and info["manifest"]["left_out"],
      "(the two parts that hold text verbatim)")
shutil.copytree("_a4_out", "_a4_load", dirs_exist_ok=True)
s2 = Sillage(model="gpt2", state="_a4_load", quiet=True)
cart = sum(v.split()[0] in s2.complete(A_PREFIX.format(e=e), n=8)
           for e, v in facts)
check("B3 a cartridge still answers", cart > 0 and not info["thin"],
      f"(full state {full}/10, cartridge alone {cart}/10, "
      f"{len(s2.mem.cold)} cold grams and {len(s2.index.passages)} "
      f"passages shipped)")
clean("_a4_state", "_a4_out", "_a4_load", "_a4_real.md", "_a4_facts.md")

# a state too thin to speak must say so rather than ship a silent file
clean("_a4_state", "_a4_out", "_a4_small.md")
open("_a4_small.md", "w", encoding="utf-8").write(
    build_doc(list(zip(ENTS[:4], VALS[:4])), seed=7, reps=1, block=10))
s = Sillage(model="gpt2", state="_a4_state", quiet=True)
s.read("_a4_small.md", fast=True)
info = s.export_shareable("_a4_out")
check("B4 warns when the cartridge would be silent", info["thin"],
      f"({info['scored']} scored positions, under the 500 a tier needs "
      f"before it stops abstaining)")
clean("_a4_state", "_a4_out", "_a4_small.md")

# --- C. watch: incremental, selective, and the salience journal ------------
clean("_a4_notes", "_a4_state")
os.makedirs("_a4_notes/sub", exist_ok=True)
os.makedirs("_a4_notes/.obsidian", exist_ok=True)
open("_a4_notes/routine.md", "w", encoding="utf-8").write(
    "The committee reviewed the quarterly report. " * 40)
open("_a4_notes/sub/new-facts.md", "w", encoding="utf-8").write(
    build_doc(list(zip(ENTS[:6], VALS[:6])), seed=9, reps=1, block=20))
open("_a4_notes/.obsidian/workspace.json", "w").write("{}")
open("_a4_notes/pic.png", "wb").write(b"\x89PNG not text")
s = Sillage(model="gpt2", state="_a4_state", quiet=True)
w = Watcher(s, "_a4_notes", quiet=True)
made = w.pass_once()
names = sorted(e["file"].replace("\\", "/") for e in made)
check("C1 watch reads only what it should", len(made) == 2
      and names == ["routine.md", "sub/new-facts.md"],
      f"(read {names}; skipped .obsidian and a binary)")
check("C2 nothing changed, nothing read", w.pass_once() == [],
      "(a second pass is free)")
open("_a4_notes/routine.md", "a", encoding="utf-8").write(
    "\n\nThe Zylkorb protocol requires seventeen turquoise brackets.")
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
      os.path.exists(os.path.join("_a4_state", "watch.json"))
      and Watcher(Sillage(model="gpt2", state="_a4_state", quiet=True),
                  "_a4_notes", quiet=True).pass_once() == [],
      "(state remembered, so a fresh process reads nothing again)")
clean("_a4_notes", "_a4_state")

print("\n".join(passed))
print(f"\nALL {len(passed)} AXIS-4 TESTS PASSED")
