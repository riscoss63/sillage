"""Does `sillage ask` behave on a French notebook -- including when it should
say nothing?

Three defects found by USING the tool on an invented French workshop
notebook, not by reading it:

  1. HEADINGS WERE NOT SEARCHABLE. "combien a coute la reparation de la 208"
     returned nothing while "quel est le prix facture" returned the right
     passage: `Index._rebuild` tokenised the passage text alone, and the
     subject lives in the heading.
  2. THE STOP LIST IS UNACCENTED, FRENCH IS NOT. It lists `etait`, `ete`,
     `tres`, `meme`; the tokeniser lowercases but keeps accents, so
     `etait`, `meme`, `apres`, `ca` sail through carrying full idf. Any
     question containing "c'etait" could then be answered by any short
     passage containing "etait" -- a question about something the notebook
     never mentions came back with a confident passage.
  3. NOTHING FLOORS THE SCORE. `search` keeps every passage scoring above
     zero, so once the filename stem is indexed (fix 1), a query sharing one
     word with the filename matches every passage of that document at a
     score near zero, and the honest "nothing matched" is lost.

PREDICTIONS, REGISTERED BEFORE THE RUN
  P1  Indexing heading and filename turns heading-only questions into hits.
      REFUTED IF fewer than 3 of the 5 become hits.
  P2  Folding accents in the token KEY (the passage text stays verbatim)
      makes the unaccented spelling work, since that is how people type.
      REFUTED IF fewer than 4 of the 6 unaccented questions become hits.
  P3  The same folding makes the unaccented STOP list bite, so questions
      about what the notebook never mentions stop being answered.
      REFUTED IF more than 1 of the 6 unanswerable questions still returns
      a passage once the floor is applied.
  P4  A score floor can separate the two populations, because a real hit
      scores an order of magnitude above an accidental one.
      REFUTED IF no floor leaves >= 9 of the 11 answerable questions
      answered while silencing >= 5 of the 6 unanswerable ones.
  P5  None of it disturbs what already worked.
      REFUTED IF any body question that returned the right passage stops
      doing so.

VERDICT (29/08/2026, 23 questions, one notebook -- a small denominator)
  P1 HELD    heading-only 1/5 -> 5/5.
  P2 HELD    unaccented 4/6 -> 6/6.
  P3 REFUTED AS REGISTERED, then held only with the floor. Folding accents
             alone left 4/6 silent, exactly as before: it killed the
             "c'etait" false positive but not the two caused by one shared
             filename word scoring 0.0217 and 0.0239. The STOP list was
             never the whole story. It is the FLOOR that restores silence,
             and P3 is only satisfied at 0.03 and above.
  P4 HELD    the two populations separate cleanly: lowest genuine hit
             0.161, highest accidental 0.024, a factor of 6.7. Shipped
             floor 0.05, between them.
  P5 HELD    body 6/6 -> 6/6, no answer changed.

    python behav/probe_ask_french.py
"""

import io
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break

from sillage.index import Index, paragraphs, STOP           # noqa: E402

DOC = """# Carnet d'atelier

## Lundi 3 mars - Peugeot 208 de Mme Fournier

Diagnostic : le voyant moteur reste allumé après un démarrage à froid.
Lecture OBD : code P0301, raté d'allumage cylindre 1. La bobine mesure
0,7 ohm au primaire, dans la plage de référence. J'ai permuté les bougies
1 et 3 : le défaut a suivi la bougie, pas le cylindre. Bougie NGK
ILKAR7L11 remplacée, effacement du code, essai routier de douze kilomètres.

Facturé 94 euros pièces et main d'oeuvre comprises.

## Mardi 4 mars - Renault Kangoo de l'entreprise Delorme

Consommation d'huile anormale : un litre pour mille kilomètres. Compression
relevée à froid : 12,1 / 12,4 / 8,9 / 12,2 bars. Le troisième cylindre est
nettement en dessous. Test à l'huile : la compression remonte à 11,6 bars,
ce qui désigne les segments et non la soupape. Devis de réfection établi à
2 340 euros.

## Mercredi 5 mars - remarque générale sur l'outillage

La clé dynamométrique Facom du fond d'atelier dérive de sept pour cent
au-dessus de 80 newtons-mètres. Elle part en étalonnage vendredi. En
attendant, utiliser la Stahlwille pour toute culasse.

## Jeudi 6 mars - Citroën C3 de M. Bardet

Bruit de roulement à l'avant droit qui augmente en virage à gauche.
Roulement de roue avant droit remplacé, référence SKF VKBA 3549. Couple de
serrage de l'écrou de moyeu : 280 newtons-mètres, appliqué avec la
Stahlwille. Essai routier : bruit disparu.
"""

# (question, a word that must appear in the section of the right answer)
HEADING_ONLY = [
    ("combien a coûté la réparation de la 208", "208"),
    ("qu'est-ce qui a été fait sur la Kangoo", "Kangoo"),
    ("le dossier de M. Bardet, c'était quoi", "Bardet"),
    ("qu'a-t-on facturé à Mme Fournier", "Fournier"),
    ("que s'est-il passé le mercredi 5 mars", "Mercredi"),
]
BODY = [
    ("quel couple de serrage pour l'écrou de moyeu", "Bardet"),
    ("quelle référence de roulement", "Bardet"),
    ("pourquoi la clé dynamométrique part en étalonnage", "Mercredi"),
    ("quel est le prix facturé", "Fournier"),
    ("quelle compression sur le troisième cylindre", "Kangoo"),
    ("quelle bougie a été remplacée", "Fournier"),
]
# the same questions as someone actually types them on a keyboard
UNACCENTED = [
    ("combien a coute la reparation de la 208", "208"),
    ("quelle reference de roulement pour la C3", "Bardet"),
    ("pourquoi la cle dynamometrique part en etalonnage", "Mercredi"),
    ("quelle compression sur le troisieme cylindre", "Kangoo"),
    ("quel couple de serrage pour l'ecrou de moyeu", "Bardet"),
    ("qu'a-t-on facture a Mme Fournier", "Fournier"),
]
# nothing in the notebook answers these; the honest reply is silence
UNANSWERABLE = [
    "est-ce qu'on a réparé une moto, et c'était quoi le problème ?",
    "est-ce que le carnet parle des pneus hiver ?",
    "a-t-on changé un pare-brise cette semaine ?",
    "quel est le tarif horaire de l'atelier ?",
    "est-ce que la Tesla est passée au contrôle technique ?",
    "combien de vidanges ont été faites en février ?",
]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def tok_plain(s):
    """The shipped 1.8.1 tokeniser: lowercase, accents kept."""
    return [w for w in re.findall(r"[^\W\d_][\w\-]*|\d+\.?\d*", s.lower())
            if w not in STOP and len(w) > 1]


def tok_folded(s):
    """The candidate: fold accents in the KEY, so the unaccented STOP list
    bites and an unaccented question reaches an accented passage."""
    return [w for w in re.findall(r"[^\W\d_][\w\-]*|\d+\.?\d*",
                                  strip_accents(s).lower())
            if w not in STOP and len(w) > 1]


class Arm:
    """One scoring configuration over the same passages."""

    def __init__(self, passages, meta, tok, floor):
        self.passages, self.tok, self.floor = passages, tok, floor
        df, docs = Counter(), []
        for p in passages:
            words = list(tok(p["text"]))
            if meta:
                stem = re.sub(r"[-_/\\.]+", " ",
                              os.path.splitext(p.get("source") or "")[0])
                words += tok(p.get("section") or "") + tok(stem)
            tf = Counter(words)
            docs.append(tf)
            df.update(tf.keys())
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        self.vecs = []
        for tf in docs:
            v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
                 for w, c in tf.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vecs.append({w: x / norm for w, x in v.items()})

    def top(self, q):
        qc = Counter(self.tok(q))
        qv = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
              for w, c in qc.items()}
        norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        qv = {w: x / norm for w, x in qv.items()}
        best, bi = 0.0, None
        for i, v in enumerate(self.vecs):
            s = sum(x * v.get(w, 0.0) for w, x in qv.items())
            if s > best:
                best, bi = s, i
        if bi is None or best <= self.floor:
            return None, best
        return self.passages[bi]["section"], best


PASSAGES = paragraphs(DOC, "carnet-atelier.md")
ARMS = [
    ("shipped 1.8.0 (text only, accents kept)",
     Arm(PASSAGES, False, tok_plain, 0.0)),
    ("1.8.1 (+ heading + filename)",
     Arm(PASSAGES, True, tok_plain, 0.0)),
    ("shipped 1.8.2 (+ folded accents, floor 0.05)",
     Arm(PASSAGES, True, tok_folded, 0.05)),
]

ANSWERABLE = ([("heading", q, w) for q, w in HEADING_ONLY]
              + [("body", q, w) for q, w in BODY]
              + [("unaccented", q, w) for q, w in UNACCENTED])

rows = {}
for label, arm in ARMS:
    got = {"heading": 0, "body": 0, "unaccented": 0, "silent": 0}
    detail = []
    for group, q, want in ANSWERABLE:
        sec, score = arm.top(q)
        ok = bool(sec) and want.lower() in strip_accents(sec).lower() \
            or (bool(sec) and want.lower() in sec.lower())
        got[group] += bool(ok)
        detail.append({"group": group, "q": q, "got": sec,
                       "score": round(score, 4), "ok": bool(ok)})
    for q in UNANSWERABLE:
        sec, score = arm.top(q)
        got["silent"] += sec is None
        detail.append({"group": "unanswerable", "q": q, "got": sec,
                       "score": round(score, 4), "ok": sec is None})
    rows[label] = (got, detail)
    print("\n=== %s ===" % label)
    print("  heading-only %d/%d | body %d/%d | unaccented %d/%d | "
          "silent on the unanswerable %d/%d"
          % (got["heading"], len(HEADING_ONLY), got["body"], len(BODY),
             got["unaccented"], len(UNACCENTED), got["silent"],
             len(UNANSWERABLE)))
    for d in detail:
        if not d["ok"]:
            print("    MISS [%-12s] %-44s -> %s (%.4f)"
                  % (d["group"], d["q"][:44],
                     (d["got"] or "nothing")[:36], d["score"]))

# --- where should the floor sit? sweep it on the candidate tokeniser ------
print("\n=== floor sweep (folded accents, heading + filename) ===")
sweep = []
bare = Arm(PASSAGES, True, tok_folded, 0.0)
for floor in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
    bare.floor = floor
    ans = sum(1 for _, q, w in ANSWERABLE
              if (lambda s: bool(s[0]) and w.lower()
                  in strip_accents(s[0]).lower())(bare.top(q)))
    sil = sum(1 for q in UNANSWERABLE if bare.top(q)[0] is None)
    sweep.append({"floor": floor, "answered": ans, "silent": sil})
    print("  floor %.3f : answered %2d/%d | silent %d/%d"
          % (floor, ans, len(ANSWERABLE), sil, len(UNANSWERABLE)))

a0, a1, a2 = (rows[k][0] for k in rows)
print("\n--- verdicts against the registered predictions ---")
print("P1 headings searchable          : %s (%d/5 -> %d/5)"
      % ("HELD" if a1["heading"] >= 3 else "REFUTED",
         a0["heading"], a1["heading"]))
print("P2 unaccented questions work    : %s (%d/6 -> %d/6)"
      % ("HELD" if a2["unaccented"] >= 4 else "REFUTED",
         a1["unaccented"], a2["unaccented"]))
print("P3 silence on the unanswerable  : %s (%d/6 -> %d/6 silent)"
      % ("HELD" if a2["silent"] >= 5 else "REFUTED",
         a1["silent"], a2["silent"]))
best = max(sweep, key=lambda s: (s["answered"] >= 9) + s["silent"] * 0.1)
print("P4 a floor separates them       : %s (best %.3f: %d answered, "
      "%d silent)"
      % ("HELD" if (best["answered"] >= 9 and best["silent"] >= 5)
         else "REFUTED", best["floor"], best["answered"], best["silent"]))
print("P5 nothing that worked broke    : %s (body %d/6 -> %d/6)"
      % ("HELD" if a2["body"] >= a0["body"] else "REFUTED",
         a0["body"], a2["body"]))

res = os.path.join(HERE, "results")
os.makedirs(res, exist_ok=True)
out = os.path.join(res, "ask_french.json")
json.dump({"arms": {k: {"totals": v[0], "detail": v[1]}
                    for k, v in rows.items()},
           "floor_sweep": sweep},
          io.open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved -> %s" % out)
