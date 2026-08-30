"""Does the memory get WORSE at document A as you read documents B, C, D?

This is the scaling question, and there is now a mechanism to suspect:
the v1 semantic tier centres its keys on `mu`, a running mean over
every hidden state the memory has ever scored. Reading a second
document folds that document's states into the centre used to key the
first one. A scoring pass of 182 tokens of unrelated prose was already
enough to turn `Brindas Kolvec` into `Brigitte Lefevre`.

Reading is the legitimate path -- a user reads more notes every day --
so measure it directly rather than through a probe helper.

Registered BEFORE the run:

  X1  Recall of the eight facts of document A falls as unrelated
      documents are read after it.
      FALSIFIED if recall is flat across all four checkpoints.
  X2  The loss is attributable to the semantic tier: with `--no-semantic`
      the same schedule shows no fall (it starts lower and stays).
      FALSIFIED if the no-semantic arm falls too -- then the cause is
      the shared n-gram matrix filling up, not the centre.
  X3  The cold store still HOLDS the lost facts: following its own
      successors from the question's key still spells the answer.
      FALSIFIED if the store has lost them -- that would be capacity,
      not arbitration.
"""
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sillage.core as core                          # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from probe_readout_dial import DOC, ANSWERABLE       # noqa: E402
from probe_reflow import reflow                      # noqa: E402
from probe_lamc import chain                         # noqa: E402
from probe_whymiss import Q                          # noqa: E402

OTHERS = {
    "pain": """La cuisson du pain au levain demande une attention constante a
la temperature de la piece. Un levain jeune double de volume en quatre a six
heures a vingt-quatre degres, mais il lui faut le double de temps dans une
cuisine fraiche d'hiver. Le petrissage se fait en deux temps, avec un repos
d'une demi-heure entre les deux, ce qui laisse le gluten se detendre. La
farine complete boit davantage d'eau que la farine blanche, et il faut donc
augmenter l'hydratation d'environ cinq pour cent quand on en met un tiers
dans le melange. La cuisson demarre a four tres chaud, avec de la vapeur
pendant les vingt premieres minutes, puis se poursuit a chaleur moderee.""",
    "jardin": """Le potager d'automne se prepare des la fin du mois d'aout.
Les semis de mache et d'epinard se font en place, a un centimetre de
profondeur, sur une terre finement emiettee et maintenue humide. Les poireaux
repiques en juillet demandent un buttage regulier pour blanchir les futs. Il
faut arracher les pieds de tomate des les premieres nuits fraiches, car le
mildiou gagne vite par temps humide. Le compost de printemps se retourne une
derniere fois avant les gelees, et l'on couvre les planches nues d'un paillis
de feuilles mortes pour proteger la vie du sol pendant l'hiver.""",
    "velo": """L'entretien d'un velo de route tient en quelques gestes
reguliers. La chaine se degraisse et se relubrifie tous les trois cents
kilometres, davantage par temps de pluie. Les patins d'un frein sur jante
s'usent en biais si l'etrier est mal centre, et il faut alors reprendre le
reglage a la vis laterale. La pression des pneus se verifie avant chaque
sortie : elle chute naturellement de l'ordre d'un demi-bar par semaine. Un
jeu dans le pedalier s'entend avant de se voir, et se corrige en reprenant la
precharge du roulement avant de resserrer les vis de la manivelle.""",
}


def recall(s):
    hits, miss = 0, []
    for prompt, want in ANSWERABLE:
        txt = s.complete(prompt, n=12, temp=0.0)
        if want.lower() in txt.lower():
            hits += 1
        else:
            miss.append((want, txt.strip()[:34]))
    return hits, miss


def arm(label, semantic):
    tmp = tempfile.mkdtemp(prefix="moredocs_")
    try:
        s = Sillage(model="qwen", state=tmp, semantic=semantic, quiet=True)
        for _ in range(2):
            s.read_text(reflow(DOC))
        tokr = s.load_tokenizer()
        steps = []
        h, m = recall(s)
        steps.append({"after": "A only", "recall": h, "mu_n": s.mem.mu_n,
                      "grams": len(s.mem.cold), "miss": m})
        print(f"  [{label}] A only            recall {h}/8  mu_n "
              f"{s.mem.mu_n}  grams {len(s.mem.cold)}", flush=True)
        for name, text in OTHERS.items():
            s.read_text(reflow(text))
            h, m = recall(s)
            steps.append({"after": name, "recall": h, "mu_n": s.mem.mu_n,
                          "grams": len(s.mem.cold), "miss": m})
            print(f"  [{label}] + read '{name}'    recall {h}/8  mu_n "
                  f"{s.mem.mu_n}  grams {len(s.mem.cold)}"
                  + (f"   lost: {[x[0] for x in m]}" if m else ""),
                  flush=True)
        ch = "".join(chain(s.mem, tokr, Q))
        print(f"  [{label}] cold-store chain at the end: {ch!r}", flush=True)
        return {"steps": steps, "chain_at_end": ch}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("read the report, then read three unrelated documents:")
res = {"semantic_on": arm("sem ON ", None),
       "semantic_off": arm("sem OFF", False)}
on = [x["recall"] for x in res["semantic_on"]["steps"]]
off = [x["recall"] for x in res["semantic_off"]["steps"]]
res["verdict"] = {
    "X1": {"recall_curve_on": on, "holds": min(on) < on[0]},
    "X2": {"recall_curve_off": off, "off_falls": min(off) < off[0]},
    "X3": {"chain": res["semantic_on"]["chain_at_end"],
           "holds": "Brindas" in res["semantic_on"]["chain_at_end"]}}
print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "moredocs.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with io.open(out, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1, ensure_ascii=False)
print(f"written {out}")
