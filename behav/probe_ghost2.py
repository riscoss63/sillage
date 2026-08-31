"""Two things `probe_ghost` left open, and both decide the product.

It measured 0.043 ms, 100% coverage and 92.5% verbatim on the very text
the memory had read twice. Two objections to that, and this answers
both.

FIRST, the 7.5%. A suggestion that is not the continuation at THIS
position may still be text the document contains somewhere else: a
4-gram that appears twice with different continuations makes the chain
follow the more frequent one. That is a very different failure from
inventing, and the product claim -- "it can only emit what it read" --
stands or falls on it.

SECOND, the 100%. Coverage was measured by replaying the exact tokens
the memory had just read twice, so every gram was present by
construction. Nobody retypes their notes verbatim. What matters is a
text on the SAME subject, in the same voice, that is not the same text.

Registered BEFORE the run:

  J1  Every suggestion, verbatim-at-this-position or not, is a substring
      of what was read. FALSIFIED by a single counter-example -- and
      that single counter-example would sink the claim.
  J2  Coverage on same-subject, different text is between 10% and 60%.
      Below 10% the feature is dead weight; above 60% would mean the
      fixture is too close to the original to be a fair test.
  J3  On that text, suggestions that ARE the right continuation stay
      the majority of what is offered (>= 50%).
"""
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sillage.core as core                          # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from probe_readout_dial import DOC as DOC_FR         # noqa: E402
from probe_ghost import ghost                        # noqa: E402

# Same station, same voice, same vocabulary -- a LATER visit. This is
# what a person actually types after the memory has read the first
# report: the subject repeats, the sentences do not.
FOLLOW_UP = """Rapport d'intervention 2026-207 - Station de pompage de Vernouil

La visite de controle a ete conduite le 3 octobre 2026 par l'equipe de
maintenance sur la station de pompage de Vernouil, en presence du
responsable d'exploitation.

Le groupe motopompe numero 3 presente un jeu radial de 0,19 millimetre au
palier avant, contre 0,42 millimetre lors de l'intervention precedente. La
garniture mecanique posee en juin ne montre aucune fuite. Le couple de
serrage applique aux huit goujons de bride est de 62 newtons-metres,
inchange.

La pression de refoulement mesuree s'etablit a 4,6 bars pour un debit de
115 metres cubes par heure. La temperature du palier arriere se maintient
a 53 degres.

Le filtre amont a ete nettoye et son encrassement estime a 25 pour cent de
la section utile. Le compteur horaire du groupe numero 3 affiche 23 940
heures de fonctionnement.

Le clapet anti-retour de la conduite secondaire a ete remplace, ce que le
rapport precedent recommandait sans caractere d'urgence.
"""


def main():
    tmp = tempfile.mkdtemp(prefix="ghost2_")
    res = {}
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        source = Sillage.reflow(DOC_FR)
        for _ in range(2):
            s.read_text(source)
        tok = s.load_tokenizer()
        mem = s.mem

        # J1: is every suggestion a substring of what was read?
        ids_src = tok.encode(source)
        src_txt = tok.decode(ids_src)
        invented = []
        checked = 0
        for i in range(core.NGRAM, len(ids_src) - 10, 2):
            g = ghost(mem, ids_src[:i])
            if len(g) < 2:
                continue
            checked += 1
            if tok.decode(g) not in src_txt:
                invented.append({"at": i, "text": tok.decode(g)})
        res["J1"] = {"checked": checked, "not_in_source": len(invented),
                     "examples": invented[:3],
                     "holds": not invented}
        print(f"J1  {checked} suggestions checked, "
              f"{len(invented)} not present in what was read", flush=True)

        # J2/J3: a later report on the same station, never read
        ids_new = tok.encode(Sillage.reflow(FOLLOW_UP))
        offered = right = 0
        lens = []
        shown = []
        for i in range(core.NGRAM, len(ids_new) - 10, 2):
            g = ghost(mem, ids_new[:i])
            if len(g) < 2:
                continue
            offered += 1
            lens.append(len(g))
            ok = g == list(ids_new[i:i + len(g)])
            right += ok
            if len(shown) < 6:
                shown.append({"typed": tok.decode(ids_new[max(0, i - 10):i]),
                              "ghost": tok.decode(g),
                              "correct": bool(ok)})
        n = len(range(core.NGRAM, len(ids_new) - 10, 2))
        res["J2"] = {"positions": n, "offered": offered,
                     "coverage": round(offered / max(1, n), 3),
                     "mean_len": round(float(np.mean(lens)), 1) if lens else 0,
                     "holds": 0.10 <= offered / max(1, n) <= 0.60}
        res["J3"] = {"right": right, "of_offered": offered,
                     "share": round(right / max(1, offered), 3),
                     "holds": right / max(1, offered) >= 0.50}
        res["samples"] = shown
        print(f"J2  coverage on a LATER report of the same station: "
              f"{res['J2']['coverage']:.0%} ({offered}/{n}), mean "
              f"{res['J2']['mean_len']} tokens", flush=True)
        print(f"J3  of those, {right} were the right continuation "
              f"({res['J3']['share']:.0%})", flush=True)
        print("", flush=True)
        for smp in shown:
            print(f"  {'OK ' if smp['correct'] else '.. '}"
                  f"...{smp['typed']!r} -> {smp['ghost']!r}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + json.dumps({k: res[k] for k in ("J1", "J2", "J3")},
                            indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "ghost2.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
