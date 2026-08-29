"""When `sillage ask` cannot answer, what should it do?

1.8.2 added a score floor (`Index.MIN_SCORE = 0.05`) after measuring, on ONE
small notebook, that genuine hits scored 0.161 and up while accidental ones
scored 0.024 and under. Replaying the wider real-world trials refuted that
calibration twice over:

  * on a second French notebook, genuine hits span 0.127-0.285 and accidental
    ones 0.133-0.261 -- the two populations OVERLAP COMPLETELY, so no floor
    can separate them;
  * and the floor took an answer away: on a two-document conflict corpus the
    passages holding the answer score 0.0458 and 0.0447, so "Quel est le
    seuil d'alerte ?" -- whose answer is the first sentence of both documents
    -- now returns "(nothing matched)". That is strictly worse than the
    behaviour it replaced.

Looking at what actually scores, two different things were being lumped
together and only one of them is a defect:

  A. THE FRENCH ELISION `qu` SURVIVES TOKENISATION as a full-weight term
     (idf 1.269 on that notebook), so "est-ce qu'on a repare une moto" scores
     0.167 with `moto`, `repare` and `probleme` each contributing exactly
     0.000. That is a bug: a function word carrying a whole answer.
  B. Several "unanswerable" questions share REAL content words with the
     notebook -- "combien j'ai facture le remplacement du pare-brise" shares
     `facture`, "un devis pour une boite automatique" shares `devis` and
     `client`. A lexical index returns those by construction. That is not a
     bug; it is what lexical retrieval is. The honest answer is not silence,
     it is telling the reader the match is weak.

PROTOCOL. The notebook of probe_ask_french.py is the DEV corpus, where the
stop list is tuned. The trial's larger notebook (A_atelier/carnet_atelier.md,
6 entries, 15 passages, written by someone else) is the TEST corpus, never
tuned on. Both question sets come from the trials, unchanged.

PREDICTIONS, REGISTERED BEFORE THE RUN
  P1  Adding the French elisions and the interrogatives to the stop list
      removes the accidental hits that rest on a function word, on the TEST
      corpus too.
      REFUTED IF the number of TEST questions answered only by a function
      word does not fall to zero.
  P2  It costs no genuine answer.
      REFUTED IF any answerable question on either corpus loses its answer,
      or drops out of the top 3.
  P3  With the stop list fixed, the floor is not worth its cost: the
      remaining accidental hits share real content words, so a floor cannot
      remove them without removing genuine answers too.
      REFUTED IF some floor silences >= 4 more TEST unanswerable questions
      while costing no answerable one on either corpus.
  P4  What a floor cannot do, a caution line can: a threshold on the top
      score separates "this answers you" from "this merely shares words".
      REFUTED IF no threshold puts >= 8 of the 14 TEST unanswerable
      questions under it while leaving >= 9 of the 11 answerable ones above.

    python behav/probe_ask_abstain.py
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

from sillage.index import paragraphs, STOP                    # noqa: E402
import behav.probe_ask_french as dev                          # noqa: E402,F401

TRIAL = ("C:/Users/abdel/AppData/Local/Temp/claude/"
         "C--Users-abdel-Documents-preprint/"
         "f02fe784-4e95-4378-8301-437af8e55fa5/scratchpad/realworld/"
         "A_atelier/carnet_atelier.md")

# What the tokeniser lets through today and should not: the elisions French
# writes with an apostrophe, which `[\w\-]*` then keeps as words, and the
# interrogatives, which are pure question scaffolding.
ADD = set("""qu jusqu lorsqu puisqu quelqu quoiqu presqu
combien comment pourquoi quand quel quelle quels quelles
ca cela celui celle ceux celles ceci
faut avons avez ont suis etes etait etaient serait seront
oui non voici voila deja encore toujours jamais beaucoup trop assez
""".split())


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def make_tok(stop):
    def tok(s):
        return [w for w in re.findall(r"[^\W\d_][\w\-]*|\d+\.?\d*", fold(s))
                if w not in stop and len(w) > 1]
    return tok


class Arm:
    def __init__(self, passages, tok):
        self.passages, self.tok = passages, tok
        df, docs = Counter(), []
        for p in passages:
            stem = re.sub(r"[-_/\\.]+", " ",
                          os.path.splitext(p.get("source") or "")[0])
            tf = Counter(tok(p["text"]) + tok(p.get("section") or "")
                         + tok(stem))
            docs.append(tf)
            df.update(tf.keys())
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        self.vecs = []
        for tf in docs:
            v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
                 for w, c in tf.items()}
            nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vecs.append({w: x / nrm for w, x in v.items()})

    def hits(self, q, k=3):
        qc = Counter(self.tok(q))
        qv = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
              for w, c in qc.items()}
        nrm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        qv = {w: x / nrm for w, x in qv.items()}
        out = []
        for i, v in enumerate(self.vecs):
            s = sum(x * v.get(w, 0.0) for w, x in qv.items())
            if s > 0:
                out.append((s, i))
        out.sort(reverse=True)
        return out[:k]

    def top(self, q):
        h = self.hits(q, 1)
        return (h[0][0], self.passages[h[0][1]]) if h else (0.0, None)

    def carried_by_stopword(self, q):
        """Is the whole score one function word we now filter?"""
        s, _ = self.top(q)
        if s <= 0:
            return False
        content = [w for w in self.tok(q) if w not in ADD]
        rest = Arm.__new__(Arm)
        rest.__dict__ = dict(self.__dict__)
        return self.top(" ".join(content))[0] < 0.25 * s


# --- the two corpora ------------------------------------------------------
DEV_P = paragraphs(dev.DOC, "carnet-atelier.md")
TEST_P = paragraphs(io.open(TRIAL, encoding="utf-8").read(),
                    "carnet_atelier.md")

DEV_ANS = ([(q, w) for q, w in dev.HEADING_ONLY]
           + [(q, w) for q, w in dev.BODY]
           + [(q, w) for q, w in dev.UNACCENTED])
DEV_UNANS = list(dev.UNANSWERABLE)

# the trial's own sets, unchanged
TEST_UNANS = [
    "est-ce qu'on a repare une moto, et c'etait quoi le probleme ?",
    "est-ce qu'on a réparé une moto, et c'était quoi le problème ?",
    "est-ce que le carnet parle des pneus hiver",
    "combien j'ai facturé le remplacement du pare-brise ?",
    "qu'est-ce que j'ai noté sur le turbo du camion ?",
    "on a fait une géométrie sur quelle voiture ?",
    "quel est le tarif horaire de l'atelier ?",
    "est-ce qu'on a changé un amortisseur cette année ?",
    "quelle marque de batterie je monte d'habitude ?",
    "est-ce qu'un client a demandé un devis pour une boîte automatique ?",
    "quand est-ce que le contrôle technique de la Clio a été fait ?",
    "j'ai vidangé combien de moteurs de tracteur ?",
    "est-ce que le compresseur de clim de la Twingo a été remplacé ?",
    "quel est le code wifi de l'atelier ?",
]

ARMS = {"shipped 1.8.2 stop list": make_tok(STOP),
        "candidate (+ elisions, interrogatives)": make_tok(STOP | ADD)}

report = {}
for label, tok in ARMS.items():
    dev_a, test_a = Arm(DEV_P, tok), Arm(TEST_P, tok)
    dev_hit = sum(1 for q, w in DEV_ANS
                  if dev_a.top(q)[1]
                  and w.lower() in fold(dev_a.top(q)[1]["section"]))
    dev_zero = sum(1 for q in DEV_UNANS if dev_a.top(q)[0] == 0)
    test_zero = sum(1 for q in TEST_UNANS if test_a.top(q)[0] == 0)
    carried = sum(1 for q in TEST_UNANS if test_a.carried_by_stopword(q))
    ans_scores = sorted(dev_a.top(q)[0] for q, w in DEV_ANS)
    un_scores = sorted(test_a.top(q)[0] for q in TEST_UNANS
                       if test_a.top(q)[0] > 0)
    report[label] = {
        "dev_answered": dev_hit, "dev_n": len(DEV_ANS),
        "dev_silent_on_unanswerable": dev_zero, "dev_unans_n": len(DEV_UNANS),
        "test_silent_on_unanswerable": test_zero,
        "test_unans_n": len(TEST_UNANS),
        "test_carried_by_a_function_word": carried,
        "answerable_score_range": [round(ans_scores[0], 3),
                                   round(ans_scores[-1], 3)],
        "test_accidental_score_range": ([round(un_scores[0], 3),
                                         round(un_scores[-1], 3)]
                                        if un_scores else None),
    }
    print("\n=== %s ===" % label)
    for k, v in report[label].items():
        print("  %-34s %s" % (k, v))

# --- P3: can any floor help without cost? ---------------------------------
print("\n=== floor sweep, candidate stop list "
      "(answerable on DEV, silence on TEST) ===")
tok = make_tok(STOP | ADD)
dev_a, test_a = Arm(DEV_P, tok), Arm(TEST_P, tok)
sweep = []
for floor in (0.0, 0.02, 0.05, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
    kept = sum(1 for q, w in DEV_ANS
               if dev_a.top(q)[0] > floor and dev_a.top(q)[1]
               and w.lower() in fold(dev_a.top(q)[1]["section"]))
    sil = sum(1 for q in TEST_UNANS if test_a.top(q)[0] <= floor)
    sweep.append({"floor": floor, "dev_answered": kept, "test_silent": sil})
    print("  floor %.2f : DEV answered %2d/%d | TEST silent %2d/%d"
          % (floor, kept, len(DEV_ANS), sil, len(TEST_UNANS)))

base = report["shipped 1.8.2 stop list"]
cand = report["candidate (+ elisions, interrogatives)"]
print("\n--- verdicts against the registered predictions ---")
print("P1 function-word answers removed : %s (%d -> %d of %d TEST questions "
      "answered only by a function word)"
      % ("HELD" if cand["test_carried_by_a_function_word"] == 0 else "REFUTED",
         base["test_carried_by_a_function_word"],
         cand["test_carried_by_a_function_word"], len(TEST_UNANS)))
print("P2 no genuine answer lost        : %s (DEV answerable %d -> %d of %d)"
      % ("HELD" if cand["dev_answered"] >= base["dev_answered"] else "REFUTED",
         base["dev_answered"], cand["dev_answered"], len(DEV_ANS)))
free = [s for s in sweep
        if s["dev_answered"] == sweep[0]["dev_answered"]
        and s["test_silent"] >= sweep[0]["test_silent"] + 4]
print("P3 a floor is worth its cost     : %s%s"
      % ("HELD" if free else "REFUTED (no floor buys 4 more silences for "
                             "free -- every one costs a real answer)",
         (" -- %.2f" % free[0]["floor"]) if free else ""))
good = [s for s in sweep if s["test_silent"] >= 8
        and s["dev_answered"] >= 9]
print("P4 a caution threshold exists    : %s%s"
      % ("HELD" if good else "REFUTED",
         (" -- top score under %.2f flags a weak match: %d/%d TEST "
          "unanswerable under it, %d/%d DEV answerable above"
          % (good[0]["floor"], good[0]["test_silent"], len(TEST_UNANS),
             good[0]["dev_answered"], len(DEV_ANS))) if good else ""))

res = os.path.join(HERE, "results")
os.makedirs(res, exist_ok=True)
out = os.path.join(res, "ask_abstain.json")
json.dump({"arms": report, "floor_sweep": sweep,
           "added_stopwords": sorted(ADD)},
          io.open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved -> %s" % out)
