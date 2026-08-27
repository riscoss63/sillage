"""Behavioral evaluation suite for Sillage (axe 3 / paper 6).

Six probes on invented facts (the base model cannot know them, so the
control is built in): free recall, paraphrase robustness, retention under
interference, locality on a witness text (scored WITHOUT writing),
conflict resolution across document versions, and -- by construction --
use with the supporting context absent. See NOTES_AXE3.md.

    python behavioral.py [--model gpt2] [--facts 30] [--n 8]
"""

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage  # noqa: E402

ENTS = ["Vorlagune", "Krestomil", "Zylkorb", "Marmelune", "Ilvress",
        "Quandrix", "Belfoss", "Tarnwick", "Ozmirel", "Drevkant",
        "Palverin", "Skogfeld", "Yurmalec", "Cindrovel", "Halbrix",
        "Nertoval", "Wispelgar", "Fromdahl", "Ulvestrem", "Brakkovin",
        "Selphandor", "Grimwaldt", "Tovarnell", "Exquilon", "Vandermeel",
        "Corvustag", "Blenharrow", "Astrivold", "Merrowine", "Dulcifern"]
VALS = ["turquoise llamas", "amber lanterns", "seventeen brackets",
        "copper whistles", "velvet manifests", "granite ledgers",
        "crimson pulleys", "hollow compasses", "silver bellows",
        "woven capacitors", "frozen almanacs", "painted turbines",
        "quiet magnets", "salted archives", "narrow chimneys",
        "gilded rosters", "damp lanyards", "oblique staples",
        "sturdy gondolas", "minted parasols", "braided funnels",
        "sober lighthouses", "waxed bulletins", "timid escalators",
        "plaid reservoirs", "carved dividers", "beveled sirens",
        "roasted spindles", "linen odometers", "chalky pendulums"]
ALT = ["orange baskets", "wooden flutes", "twelve hammers",
       "shallow mirrors", "dotted ribbons", "smoky kettles",
       "brittle anchors", "sandy trumpets", "pale shutters",
       "curled magnets"]

SUBJ = ["committee", "board", "council", "task force", "working group",
        "delegation", "panel", "office", "team", "department"]
VERB = ["reviewed", "discussed", "examined", "postponed", "approved",
        "rejected", "audited", "drafted", "archived", "circulated"]
OBJ = ["the quarterly report", "the budget allocation", "the hiring plan",
       "the maintenance schedule", "the safety audit", "the travel policy",
       "the vendor contract", "the training curriculum",
       "the archive migration", "the annual forecast"]

A_SENT = "The {e} protocol requires {v}."
A_PREFIX = "The {e} protocol requires"
B_PREFIX = "According to the {e} specification, the requirement is"

# Locality witness: natural prose with NO overlap with the filler templates
# or the facts -- a genuinely unrelated text, so any PPL drift is the
# memory's fault, not the corpus's.
WITNESS = """Rivers shape the land more slowly than storms, but far more
thoroughly. Over centuries a meander widens, undercuts its outer bank,
and abandons loops that become quiet oxbow lakes. Sediment carried from
distant hills settles where the current slackens, building floodplains
whose soils feed orchards and wheat. People settle along these bends for
water and trade, then spend generations defending the same bends against
the floods that made them fertile.

Bread follows a different clock. Flour, water, salt and time are enough,
yet every stage rewards patience: the slow hydration of the grain, the
long fermentation that sours and strengthens the dough, the final hour
in a hot oven when the crust sets and sings as it cools. Bakers speak of
reading the dough rather than commanding it, and the best loaves come
from mornings when nothing was hurried.

Mountains keep their own records. A glacier writes in moraines and
striations; frost writes in shattered ridgelines; forests write in the
tree line that creeps upward in warm decades and retreats in cold ones.
Walkers who return to the same valley after twenty years read the
differences the way one reads an old letter, half memory and half
surprise, and the path itself has usually moved a little as well."""


def filler(seed, sentences):
    out = []
    for k in range(sentences):
        out.append(f"The {SUBJ[(seed*31+k*7) % 10]} "
                   f"{VERB[(seed*17+k*13) % 10]} "
                   f"{OBJ[(seed*23+k*3) % 10]} on day {k+1} of "
                   f"session {seed+1}.")
    return " ".join(out)


def build_doc(facts, seed, reps=3, block=40):
    parts = []
    for r in range(reps):
        parts.append(filler(seed + r, block))
        for e, v in facts:
            parts.append(A_SENT.format(e=e, v=v))
    return "\n\n".join(parts)


def probe(s, facts, prefix_tmpl, n):
    """Fraction of facts whose value head-word appears in the completion."""
    hits, details = 0, []
    for e, v in facts:
        out = s.complete(prefix_tmpl.format(e=e), n=n)
        ok = v.split()[0] in out
        hits += ok
        details.append({"e": e, "v": v, "ok": bool(ok), "out": out[:60]})
    return hits / len(facts), details


def nll_nowrite(s, text):
    """Teacher-forced (base_ppl, memory_ppl) WITHOUT any write."""
    import torch
    tok, model = s.load_model()
    mem = s.mem
    ids = np.array(tok.encode(text), dtype=np.int64)
    n = len(ids) - 1
    mem.new_stream()
    thrG, thrS = mem.thresholds()
    need_h = mem.semantic or mem.fastweights
    nll_b = nll_m = 0.0
    cnt = 0
    x = torch.tensor(ids, device=s.device)
    a, W, S = 0, 1024, 512
    with torch.no_grad():
        while a < n:
            w = min(W, len(ids) - a)
            out = model(x[a:a + w].unsqueeze(0), output_hidden_states=need_h)
            logits = out.logits[0].float().cpu().numpy()
            mem.set_vocab(logits.shape[-1])
            hs = (out.hidden_states[-1][0].float().cpu().numpy()
                  if need_h else None)
            lo = 0 if a == 0 else W - S
            for i in range(lo, w):
                j = a + i
                if j >= n:
                    break
                truth = int(ids[j + 1])
                lb = logits[i]
                mx = lb.max()
                lp = float(lb[truth] - (mx + np.log(np.exp(lb - mx).sum())))
                la, _ = mem.adapt(lb, hs[i] if need_h else None)
                p_ad = np.exp(la - la.max())
                p_ad /= p_ad.sum()
                qG = mem.step_key(int(ids[j]))
                _, sG = mem.scores(mem.M, qG)
                sS = None
                if mem.semantic:
                    _, sS = mem.scores(mem.MS, mem.sem_key(hs[i]))
                p = mem.mix_true(float(p_ad[truth]), sG, truth, sS,
                                 mem.cold_lookup(truth), thrG, thrS)
                nll_b += -lp
                nll_m += -np.log(max(p, 1e-30))
                cnt += 1
            if a + w >= len(ids):
                break
            a += S
    return float(np.exp(nll_b / cnt)), float(np.exp(nll_m / cnt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--facts", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--state", default=os.path.join(HERE, ".behav_state"))
    a = ap.parse_args()

    rng = random.Random(7)
    facts = list(zip(ENTS[:a.facts], VALS[:a.facts]))
    changed = [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]
    witness = WITNESS

    import shutil
    shutil.rmtree(a.state, ignore_errors=True)
    s = Sillage(model=a.model, state=a.state, quiet=True)
    R = {"model": a.model, "facts": a.facts, "n": a.n}

    print("== 0. controles avant lecture (memoire vide) ==", flush=True)
    R["recall_base"], _ = probe(s, facts, A_PREFIX, a.n)
    R["para_base"], _ = probe(s, facts, B_PREFIX, a.n)
    b0, m0 = nll_nowrite(s, witness)
    R["witness_before"] = {"base": b0, "mem": m0}
    print(f"  rappel base {R['recall_base']:.0%} | paraphrase base "
          f"{R['para_base']:.0%} | temoin PPL {b0:.2f} (mem {m0:.2f})",
          flush=True)

    print("== 1-2. lecture du dossier v1, rappel + paraphrase ==",
          flush=True)
    s.read_text(build_doc(facts, seed=0), "dossier_v1")
    s.save()
    R["recall_v1"], det = probe(s, facts, A_PREFIX, a.n)
    R["para_v1"], detB = probe(s, facts, B_PREFIX, a.n)
    print(f"  rappel {R['recall_v1']:.0%} | paraphrase {R['para_v1']:.0%}",
          flush=True)

    print("== 4. localite (temoin, sans ecrire) ==", flush=True)
    b1, m1 = nll_nowrite(s, witness)
    R["witness_after"] = {"base": b1, "mem": m1}
    print(f"  temoin PPL mem {m0:.2f} -> {m1:.2f} "
          f"(delta {100*(m1-m0)/m0:+.1f}%)", flush=True)

    print("== 3. retention apres ~20k tokens d'interference ==", flush=True)
    s.read_text(filler(200, 900), "interference_1")
    s.read_text(filler(300, 900), "interference_2")
    s.save()
    R["recall_after_interf"], _ = probe(s, facts, A_PREFIX, a.n)
    print(f"  rappel {R['recall_v1']:.0%} -> "
          f"{R['recall_after_interf']:.0%}", flush=True)

    print("== 5. conflits : v2 change 10 valeurs ==", flush=True)
    doc_v2 = build_doc([(e, dict(changed).get(e, v)) for e, v in facts],
                       seed=50)
    s.read_text(doc_v2, "dossier_v2")
    s.save()
    new1, _ = probe(s, changed, A_PREFIX, a.n)
    old1, _ = probe(s, [(e, dict(facts)[e]) for e, _ in changed],
                    A_PREFIX, a.n)
    R["conflict_after_1"] = {"new": new1, "old": old1,
                             "neither": 1 - new1 - old1}
    print(f"  apres v2 x1 : nouvelle {new1:.0%} | ancienne {old1:.0%} | "
          f"confusion {1-new1-old1:.0%}", flush=True)
    s.read_text(doc_v2, "dossier_v2_bis")
    s.save()
    new2, _ = probe(s, changed, A_PREFIX, a.n)
    old2, _ = probe(s, [(e, dict(facts)[e]) for e, _ in changed],
                    A_PREFIX, a.n)
    R["conflict_after_2"] = {"new": new2, "old": old2,
                             "neither": 1 - new2 - old2}
    print(f"  apres v2 x2 : nouvelle {new2:.0%} | ancienne {old2:.0%} | "
          f"confusion {1-new2-old2:.0%}", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"behav_{a.model}.json")
    json.dump(R, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
