"""Would a light French stemmer make `sillage ask` find more, or just more noise?

The real-world trials left one retrieval failure standing that neither the
heading fix nor the stop list touches: a question whose words are morphological
variants of the notebook's. "qui me doit encore de l'argent ?" returns nothing
although the notebook says "il reste 136 euros a devoir" -- doit/devoir and
argent/euros are simply different strings. Stemming can reach the first kind
(reparation/reparer, facture/facturer, roulement/rouler) and can never reach
the second (a synonym is not a suffix).

The question this probe settles is whether the reachable half is worth the
precision it costs, measured rather than assumed. A stemmer that maps `devis`
and `devoir` to the same stem would answer a question about an estimate with a
sentence about a debt.

PROTOCOL, as in probe_ask_abstain.py: the notebook of probe_ask_french.py is
DEV; the trial's own larger notebook (A_atelier/carnet_atelier.md) is TEST and
is never tuned on. Both question sets are the trials' own.

PREDICTIONS, REGISTERED BEFORE THE RUN
  P1  A light suffix stripper answers questions that are currently missed
      because the question and the passage use different forms of the same
      word.
      REFUTED IF it answers fewer than 2 of the 4 currently-missed
      morphological questions.
  P2  It does not cost precision: no answerable question loses its answer,
      and no unanswerable question gains one.
      REFUTED IF any DEV or TEST answerable question changes answer, or if
      the count of TEST unanswerable questions returning a passage rises.
  P3  If P1 and P2 both hold it ships; if P2 fails it does not, whatever P1
      says. Recall bought with precision is not a bargain for a tool whose
      whole claim is that `ask` is the layer you can trust.

VERDICT (30/08/2026): NOT SHIPPED. P1 was refuted in the direction nobody
expected -- the stemmer answered FEWER morphological questions than the
plain tokeniser, 1 of 4 against 2 of 4. Stripping suffixes moves a query
term and a passage term independently, and two words that used to match
exactly ("remplacee" in both) can end up on different stems when one of
them sits next to a different ending. P2 held (nothing lost, no accidental
hit gained), so this is not a precision story at all: the recall it was
supposed to buy is simply not there. The two questions that fail for
morphological reasons -- doit/devoir, argent/euros -- are a synonym problem
and a suffix stripper was never going to reach them. Kept as the record of
an idea that sounded obviously right and measured wrong.

    python behav/probe_ask_stem.py
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
import behav.probe_ask_french as dev                          # noqa: E402
import behav.probe_ask_abstain as ab                          # noqa: E402

# A deliberately conservative stripper: only the endings that are pure
# inflection in French, longest first, and never below a 4-character stem.
SUFFIXES = ("issements", "issement", "ations", "ation", "ements", "ement",
            "ances", "ance", "ences", "ence", "eurs", "euse", "euses",
            "ables", "able", "ibles", "ible", "aient", "erait", "eront",
            "ees", "ee", "es", "s", "er", "ir", "re", "ant", "ent", "e")


def stem(w):
    if w.isdigit() or len(w) <= 4:
        return w
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def tok_plain(s):
    return [w for w in re.findall(r"[^\W\d_][\w\-]*|\d+\.?\d*", fold(s))
            if w not in STOP and len(w) > 1]


def tok_stem(s):
    return [stem(w) for w in tok_plain(s)]


# the four questions the trials found missed for morphological reasons
MORPH = [
    ("qui me doit encore de l'argent ?", "136 euros"),
    ("combien de reparations facturees cette semaine", "94 euros"),
    ("quelle piece a ete remplacee sur la C3", "SKF"),
    ("qu'est-ce qui a ete releve en compression", "12,1"),
]

DEV_P = paragraphs(dev.DOC, "carnet-atelier.md")
TEST_P = paragraphs(io.open(ab.TRIAL, encoding="utf-8").read(),
                    "carnet_atelier.md")
DEV_ANS = ([(q, w) for q, w in dev.HEADING_ONLY]
           + [(q, w) for q, w in dev.BODY]
           + [(q, w) for q, w in dev.UNACCENTED])

rows = {}
for label, tok in (("shipped (no stemming)", tok_plain),
                   ("candidate (light French stemmer)", tok_stem)):
    dev_a, test_a = ab.Arm(DEV_P, tok), ab.Arm(TEST_P, tok)
    answered = {}
    for q, w in DEV_ANS:
        s, p = dev_a.top(q)
        answered[q] = (p or {}).get("section")
    morph = sum(1 for q, needle in MORPH
                if dev_a.top(q)[1] and needle in dev_a.top(q)[1]["text"])
    unans_hits = sum(1 for q in ab.TEST_UNANS if test_a.top(q)[0] > 0)
    right = sum(1 for q, w in DEV_ANS
                if answered[q] and w.lower() in fold(answered[q]))
    rows[label] = {"dev_answerable_right": right, "dev_n": len(DEV_ANS),
                   "morphological_questions_answered": morph,
                   "morph_n": len(MORPH),
                   "test_unanswerable_returning_a_passage": unans_hits,
                   "test_unans_n": len(ab.TEST_UNANS),
                   "_answers": answered}
    print("\n=== %s ===" % label)
    for k, v in rows[label].items():
        if not k.startswith("_"):
            print("  %-42s %s" % (k, v))

a, b = rows["shipped (no stemming)"], rows["candidate (light French stemmer)"]
moved = [q for q in a["_answers"]
         if a["_answers"][q] != b["_answers"][q]]
print("\n--- verdicts against the registered predictions ---")
print("P1 reaches morphological variants : %s (%d/%d -> %d/%d)"
      % ("HELD" if b["morphological_questions_answered"] >= 2 else "REFUTED",
         a["morphological_questions_answered"], a["morph_n"],
         b["morphological_questions_answered"], b["morph_n"]))
p2 = (not moved
      and b["test_unanswerable_returning_a_passage"]
      <= a["test_unanswerable_returning_a_passage"])
print("P2 costs no precision             : %s%s"
      % ("HELD" if p2 else "REFUTED",
         "" if p2 else
         " -- %d answerable question(s) changed answer, TEST accidental "
         "hits %d -> %d"
         % (len(moved), a["test_unanswerable_returning_a_passage"],
            b["test_unanswerable_returning_a_passage"])))
for q in moved[:6]:
    print("     moved: %-44s %s -> %s"
          % (q[:44], a["_answers"][q], b["_answers"][q]))
print("P3 ship it                        : %s"
      % ("YES" if (p2 and b["morphological_questions_answered"] >= 2)
         else "NO -- precision is the thing `ask` is for"))

res = os.path.join(HERE, "results")
os.makedirs(res, exist_ok=True)
out = os.path.join(res, "ask_stem.json")
json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
           for k, v in rows.items()},
          io.open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("saved -> %s" % out)
