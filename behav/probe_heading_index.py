"""Is a passage findable by the heading it sits under?

Found by using the tool, not by reading it: on a French workshop notebook,
"combien a coute la reparation de la 208" returned nothing, while "quel est
le prix facture" returned the right passage. The section heading -- "Lundi
3 mars, Peugeot 208 de Mme Fournier" -- was stored and DISPLAYED with every
hit, but `Index._rebuild` tokenised only the passage text, so nothing in the
heading, and nothing in the filename, was searchable.

That is the flagship use case failing: people put the subject in the heading
and the detail underneath, so the paragraph holding the answer is often the
one paragraph that never names what it is about.

PREDICTIONS, REGISTERED BEFORE THE RUN
  P1  Indexing the section heading and the (split) filename alongside the
      passage text turns heading-only questions into hits.
      REFUTED IF fewer than 3 of the 5 heading-only questions become hits.
  P2  It does not disturb what already worked.
      REFUTED IF any question that already returned the right passage
      returns a different one afterwards.
  P3  The dilution is affordable: heading words repeat across the passages
      of a section, so their IDF falls and every score moves.
      REFUTED IF a previously correct answer loses its rank.

VERDICT (29/08/2026): P1 held, 1/5 -> 5/5. P2 held, 6/6 -> 6/6, no answer
changed. P3 held: body scores moved by at most 0.05 and no ranking flipped.
Shipped in `Index._rebuild`; guarded by T9b in test_unit.py.

    python behav/probe_heading_index.py
"""

import io
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break

from sillage.index import Index, paragraphs, tokens      # noqa: E402

# A notebook of the shape the tool is meant for: the subject in the heading,
# the detail underneath. Every name, part number and figure is invented.
DOC = """# Carnet d'atelier

## Lundi 3 mars - Peugeot 208 de Mme Fournier

Diagnostic : le voyant moteur reste allume apres un demarrage a froid.
Lecture OBD : code P0301, rate d'allumage cylindre 1. La bobine mesure
0,7 ohm au primaire, dans la plage de reference. J'ai permute les bougies
1 et 3 : le defaut a suivi la bougie, pas le cylindre. Bougie NGK
ILKAR7L11 remplacee, effacement du code, essai routier de douze kilometres.

Facture 94 euros pieces et main d'oeuvre comprises.

## Mardi 4 mars - Renault Kangoo de l'entreprise Delorme

Consommation d'huile anormale : un litre pour mille kilometres. Compression
relevee a froid : 12,1 / 12,4 / 8,9 / 12,2 bars. Le troisieme cylindre est
nettement en dessous. Test a l'huile : la compression remonte a 11,6 bars,
ce qui designe les segments et non la soupape. Devis de refection etabli a
2 340 euros.

## Mercredi 5 mars - remarque generale sur l'outillage

La cle dynamometrique Facom du fond d'atelier derive de sept pour cent
au-dessus de 80 newtons-metres. Elle part en etalonnage vendredi. En
attendant, utiliser la Stahlwille pour toute culasse.

## Jeudi 6 mars - Citroen C3 de M. Bardet

Bruit de roulement a l'avant droit qui augmente en virage a gauche.
Roulement de roue avant droit remplace, reference SKF VKBA 3549. Couple de
serrage de l'ecrou de moyeu : 280 newtons-metres, applique avec la
Stahlwille. Essai routier : bruit disparu.
"""

# (question, a word that must appear in the section of the right answer)
HEADING_ONLY = [
    ("combien a coute la reparation de la 208", "208"),
    ("qu'est-ce qui a ete fait sur la Kangoo", "Kangoo"),
    ("le dossier de M. Bardet, c'etait quoi", "Bardet"),
    ("qu'a-t-on facture a Mme Fournier", "Fournier"),
    ("que s'est-il passe le mercredi 5 mars", "Mercredi"),
]
BODY = [
    ("quel couple de serrage pour l'ecrou de moyeu", "Bardet"),
    ("quelle reference de roulement", "Bardet"),
    ("pourquoi la cle dynamometrique part en etalonnage", "Mercredi"),
    ("quel est le prix facture", "Fournier"),
    ("quelle compression sur le troisieme cylindre", "Kangoo"),
    ("quelle bougie a ete remplacee", "Fournier"),
]


def text_only(ix):
    """The pre-fix scoring: the passage text and nothing else."""
    df, docs = Counter(), []
    for p in ix.passages:
        tf = Counter(tokens(p["text"]))
        docs.append(tf)
        df.update(tf.keys())
    n = len(docs)
    ix.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
    ix.vecs = []
    for tf in docs:
        v = {w: (1 + math.log(c)) * ix.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        ix.vecs.append({w: x / norm for w, x in v.items()})


def build(pre_fix):
    ix = Index(None)
    ix.passages = paragraphs(DOC, "carnet-atelier.md")
    text_only(ix) if pre_fix else ix._rebuild()
    return ix


def run(label, ix):
    rows = []
    for group, qs in (("heading-only", HEADING_ONLY), ("body", BODY)):
        for q, want in qs:
            hits = ix.search(q, k=1)
            sec = hits[0][1]["section"] if hits else None
            score = hits[0][0] if hits else 0.0
            ok = bool(sec) and want.lower() in sec.lower()
            rows.append({"group": group, "q": q, "want": want, "got": sec,
                         "score": round(score, 3), "ok": ok})
    nh = sum(r["ok"] for r in rows if r["group"] == "heading-only")
    nb = sum(r["ok"] for r in rows if r["group"] == "body")
    print("\n=== %s ===" % label)
    for r in rows:
        print("  %s [%-12s] %-44s -> %s (%.3f)"
              % ("ok  " if r["ok"] else "MISS", r["group"], r["q"][:44],
                 (r["got"] or "nothing")[:40], r["score"]))
    print("  heading-only %d/%d | body %d/%d"
          % (nh, len(HEADING_ONLY), nb, len(BODY)))
    return rows, nh, nb


before, bh, bb = run("BEFORE -- passage text only", build(True))
after, ah, ab = run("AFTER -- text + heading + filename (shipped)",
                    build(False))

print("\n--- verdicts against the registered predictions ---")
print("P1 heading-only questions become hits : %s (%d/5 -> %d/5)"
      % ("HELD" if ah >= 3 else "REFUTED", bh, ah))
broke = [(b["q"], b["got"], a["got"]) for b, a in zip(before, after)
         if b["ok"] and not a["ok"]]
print("P2 nothing that worked broke        : %s%s"
      % ("HELD" if not broke else "REFUTED",
         "" if not broke else " -- %s" % broke))
moved = max((abs(b["score"] - a["score"]) for b, a in zip(before, after)
             if b["group"] == "body"), default=0.0)
print("P3 body answers keep their rank      : %s (largest score move %.3f, "
      "%d/%d -> %d/%d correct)"
      % ("HELD" if ab >= bb else "REFUTED", moved, bb, len(BODY), ab,
         len(BODY)))

res = os.path.join(HERE, "results")
os.makedirs(res, exist_ok=True)
out = os.path.join(res, "heading_index.json")
json.dump({"before": before, "after": after,
           "heading_only": {"before": bh, "after": ah, "n": len(HEADING_ONLY)},
           "body": {"before": bb, "after": ab, "n": len(BODY)},
           "largest_body_score_move": round(moved, 4)},
          io.open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved -> %s" % out)
