"""Right entry, wrong paragraph -- and what actually decides it.

Three rounds of real-world trials on a French workshop notebook agree on one
number and it has never moved: `sillage ask` puts the correct ENTRY at rank 1
for 10 questions out of 10, and the correct PARAGRAPH at rank 1 for 2. The
reader is handed the diagnosis when they asked the price.

The third round measured the cause instead of guessing it: 9 of the 10
questions share NO token with the body of the paragraph that answers them, so
the whole score is a heading term -- and a heading term is attached
identically to every paragraph of its entry. Nothing in the score
distinguishes them, so the tie is broken by the L2 normalisation, and the
shortest, least lexically diverse paragraph wins. No stop-list, elision or
floor change can reach that, which is exactly why it survived all three.

Two levers exist, and this probe measures both instead of assuming either:

  HEAD_W  how loudly a heading term speaks compared with a body term.
          Lowering it does NOT break the tie -- it scales every paragraph of
          the entry by the same amount -- so the prediction is that it moves
          entry ranking and does nothing for paragraphs. Worth measuring
          precisely because it is the obvious idea.
  ALPHA   pivoted length normalisation: divide by norm**ALPHA instead of by
          norm. ALPHA=1 is cosine, where short passages win; ALPHA=0 is the
          raw dot product, where long ones do. This is the lever that can
          actually reorder paragraphs within an entry.

And one leftover from withdrawing the 0.05 floor in 1.8.3: passages scoring
5.7e-05 are now served, printed as "(relevance 0.000)", and three questions
that used to get an honest "(nothing matched)" come back with 0.004 noise.
A floor is what silences that -- but it must stay well under the 0.046 and
0.045 of the conflict corpus, which is what the 0.05 floor destroyed.

PROTOCOL. DEV is probe_ask_french.py's notebook (where thresholds were tuned
before). TEST is the trials' own larger notebook with the trials' own twelve
questions and their expected section AND paragraph, verbatim from
A_atelier/sweep2.py. The conflict corpus is the third guard.

PREDICTIONS, REGISTERED BEFORE THE RUN
  P1  Damping the heading (HEAD_W < 1) does not fix paragraph ranking.
      REFUTED IF some HEAD_W raises TEST paragraph@1 by 3 or more.
  P2  Pivoted normalisation does.
      REFUTED IF no ALPHA raises TEST paragraph@1 to 5 or more of 12 while
      keeping TEST entry@1 at 10+ and DEV answerable at 17/17.
  P3  A floor at 0.01 removes the near-zero noise and touches nothing real.
      REFUTED IF it silences any answerable question on either notebook, or
      if either conflict passage (0.046, 0.045) falls under it.

VERDICT (30/08/2026)
  P1 HELD     heading damping is not the lever: paragraph@1 stays at 4/12
              from HEAD_W 1.0 down to 0.3, and below 0.5 it starts costing
              ENTRY ranking (12/12 -> 9/12 at 0.15). It scales every
              paragraph of an entry by the same factor, which was the
              reason to expect nothing, and nothing is what it does.
  P2 REFUTED, and the reason is arithmetic rather than empirical, which
              makes it worth writing down. When a query matches only the
              heading, every paragraph of the entry has the SAME numerator
              x, and the score is x / norm**ALPHA. For two paragraphs with
              norms n1 < n2 and any ALPHA > 0, n1**ALPHA < n2**ALPHA -- so
              the shorter one wins for EVERY alpha. Pivoted normalisation
              is a monotone transform of the norm, and a monotone transform
              cannot reorder a tie it did not create. Measured: paragraph@1
              is 4/12 at alpha 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35 and 0.2,
              flat to the question. No length normalisation of any exponent
              can fix this; only something that changes the NUMERATOR can
              -- retrieving at entry level and choosing within, or merging
              the paragraphs of a section when the match is heading-only.
  P3 HELD     a 0.01 floor is free: DEV 17/17 and TEST entry@1 12/12
              unchanged, both conflict passages (0.046, 0.045) still
              returned, and TEST silence 5/14 -> 7/14 -- the two questions
              it silences are the ones 1.8.3 was serving at 0.000 and
              0.004. Shipped. The 0.05 floor of 1.8.2 loses one of the two
              conflict passages, which is what made it wrong.

WHAT THIS LEAVES. The answering paragraph is at rank 1 for 4 of 12
questions and in the TOP 3 for 11 of 12 -- and 3 is the default k. So the
reader does see the answer; it is not first. Fixing that needs two-stage
retrieval (entry, then paragraph), which is a design change and not a
constant, and it is not in this release.

    python behav/probe_ask_ranking.py
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

from sillage.index import paragraphs, tokens                  # noqa: E402
import behav.probe_ask_french as dev                          # noqa: E402
import behav.probe_ask_abstain as ab                          # noqa: E402

CONFLICT = os.path.join(os.path.dirname(ab.TRIAL), "..", "E_reform")

# (question, a word of the right SECTION, a word of the right PARAGRAPH)
# verbatim from the trials' own sweep2.py
TEST_Q = [
    ("combien a coûté la réparation de la 208 ?", "Peugeot 208", "748,50"),
    ("qu'est-ce qu'on a changé sur la Kangoo ?", "Kangoo", "VK-3348"),
    ("c'était quoi le problème de M. Bardet ?", "Kangoo", "vitrifié"),
    ("qu'est-ce que j'ai fait sur la Sandero ?", "Sandero", "condenseur"),
    ("combien j'ai facturé le Ducato ?", "Ducato", "1 236"),
    ("le véhicule de Mme Vachier avait quoi ?", "Vachier", "oxydée"),
    ("qu'est-ce que j'ai changé chez M. Ozanne ?", "Ozanne", "Injecteur 3"),
    ("la Peugeot est venue pour quoi ?", "Peugeot", "Courroie"),
    ("la Dacia avait quel problème ?", "Dacia", "Circuit vide"),
    ("il y avait un Fiat, c'était pour quoi ?", "Fiat",
     "embrayage en trois ans"),
    ("qu'est-ce qu'avait le Berlingo ?", "Berlingo", "Berlingo"),
    ("le Transit avait quel souci ?", "Transit", "Transit"),
]


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


class Arm:
    """One (HEAD_W, ALPHA, FLOOR) configuration over a set of passages."""

    def __init__(self, passages, head_w=1.0, alpha=1.0, floor=0.0):
        self.passages, self.floor = passages, floor
        df, bodies, heads = Counter(), [], []
        for p in passages:
            stem = re.sub(r"[-_/\\.]+", " ",
                          os.path.splitext(p.get("source") or "")[0])
            body = Counter(tokens(p["text"]))
            head = set(tokens(p.get("section") or "") + tokens(stem))
            bodies.append(body)
            heads.append(head)
            df.update(set(body) | head)
        n = len(passages)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        self.vecs = []
        for body, head in zip(bodies, heads):
            v = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
                 for w, c in body.items()}
            for w in head:                      # a heading term, once, at
                v[w] = v.get(w, 0.0) + head_w * self.idf.get(w, 0.0)
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            norm = norm ** alpha                # pivoted: 1 = cosine, 0 = dot
            self.vecs.append({w: x / norm for w, x in v.items()})

    def hits(self, q, k=3):
        qc = Counter(tokens(q))
        qv = {w: (1 + math.log(c)) * self.idf.get(w, 0.0)
              for w, c in qc.items()}
        nrm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        qv = {w: x / nrm for w, x in qv.items()}
        out = []
        for i, v in enumerate(self.vecs):
            s = sum(x * v.get(w, 0.0) for w, x in qv.items())
            if s > self.floor:
                out.append((s, i))
        out.sort(reverse=True)
        return [(s, self.passages[i]) for s, i in out[:k]]


DEV_P = paragraphs(dev.DOC, "carnet-atelier.md")
TEST_P = paragraphs(io.open(ab.TRIAL, encoding="utf-8").read(),
                    "carnet_atelier.md")
DEV_ANS = ([(q, w) for q, w in dev.HEADING_ONLY]
           + [(q, w) for q, w in dev.BODY]
           + [(q, w) for q, w in dev.UNACCENTED])


def measure(head_w, alpha, floor=0.0):
    dev_a = Arm(DEV_P, head_w, alpha, floor)
    test_a = Arm(TEST_P, head_w, alpha, floor)
    dev_ok = sum(1 for q, w in DEV_ANS
                 for h in [dev_a.hits(q, 1)]
                 if h and w.lower() in fold(h[0][1]["section"]))
    sec1 = pas1 = pas3 = 0
    for q, sec, txt in TEST_Q:
        h = test_a.hits(q, 3)
        if not h:
            continue
        sec1 += fold(sec) in fold(h[0][1]["section"])
        pas1 += fold(txt) in fold(h[0][1]["text"])
        pas3 += any(fold(txt) in fold(p["text"]) for _, p in h)
    silent = sum(1 for q in ab.TEST_UNANS if not test_a.hits(q, 1))
    return {"dev_answerable": dev_ok, "test_entry_1": sec1,
            "test_para_1": pas1, "test_para_3": pas3,
            "test_silent": silent}


print("baseline = what 1.8.3 ships (head_w 1.0, alpha 1.0, no floor)")
base = measure(1.0, 1.0)
print("  %s\n" % base)

print("=== P1: damping the heading ===")
p1_rows = []
for hw in (1.0, 0.7, 0.5, 0.3, 0.15):
    r = measure(hw, 1.0)
    p1_rows.append(dict(head_w=hw, **r))
    print("  head_w %.2f : entry@1 %2d/12  para@1 %2d/12  para@3 %2d/12  "
          "DEV %2d/17" % (hw, r["test_entry_1"], r["test_para_1"],
                          r["test_para_3"], r["dev_answerable"]))

print("\n=== P2: pivoted length normalisation ===")
p2_rows = []
for al in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2):
    r = measure(1.0, al)
    p2_rows.append(dict(alpha=al, **r))
    print("  alpha %.2f  : entry@1 %2d/12  para@1 %2d/12  para@3 %2d/12  "
          "DEV %2d/17" % (al, r["test_entry_1"], r["test_para_1"],
                          r["test_para_3"], r["dev_answerable"]))

print("\n=== P2b: the two together ===")
best = None
grid = []
for hw in (1.0, 0.7, 0.5, 0.3):
    for al in (1.0, 0.8, 0.6, 0.5, 0.35):
        r = measure(hw, al)
        grid.append(dict(head_w=hw, alpha=al, **r))
        good = (r["dev_answerable"] == 17 and r["test_entry_1"] >= 10)
        if good and (best is None or r["test_para_1"] > best["test_para_1"]):
            best = dict(head_w=hw, alpha=al, **r)
if best:
    print("  best that keeps DEV 17/17 and entry@1 >= 10: head_w %.2f "
          "alpha %.2f -> para@1 %d/12 (para@3 %d/12)"
          % (best["head_w"], best["alpha"], best["test_para_1"],
             best["test_para_3"]))
else:
    print("  none")

print("\n=== P3: a small floor against the near-zero noise ===")
p3_rows = []
conf_p = (paragraphs(io.open(os.path.join(CONFLICT, "doc_seuil_v1.txt"),
                             encoding="utf-8").read(), "doc_seuil_v1.txt")
          + paragraphs(io.open(os.path.join(CONFLICT, "doc_seuil_v2.txt"),
                               encoding="utf-8").read(), "doc_seuil_v2.txt"))
for fl in (0.0, 0.005, 0.01, 0.02, 0.05):
    r = measure(1.0, 1.0, fl)
    conf = Arm(conf_p, 1.0, 1.0, fl).hits("Quel est le seuil d'alerte ?", 2)
    p3_rows.append(dict(floor=fl, conflict_hits=len(conf), **r))
    print("  floor %.3f : DEV %2d/17  entry@1 %2d/12  TEST silent %2d/14  "
          "conflict passages returned %d/2"
          % (fl, r["dev_answerable"], r["test_entry_1"], r["test_silent"],
             len(conf)))

print("\n--- verdicts against the registered predictions ---")
best_hw = max(p1_rows, key=lambda r: r["test_para_1"])
print("P1 heading damping is not the lever : %s (best para@1 %d/12 against "
      "the baseline's %d/12)"
      % ("HELD" if best_hw["test_para_1"] < base["test_para_1"] + 3
         else "REFUTED", best_hw["test_para_1"], base["test_para_1"]))
ok2 = [r for r in p2_rows if r["test_para_1"] >= 5
       and r["test_entry_1"] >= 10 and r["dev_answerable"] == 17]
print("P2 pivoted normalisation is        : %s%s"
      % ("HELD" if ok2 else "REFUTED",
         (" -- alpha %.2f gives para@1 %d/12"
          % (ok2[0]["alpha"], ok2[0]["test_para_1"])) if ok2 else ""))
f01 = [r for r in p3_rows if r["floor"] == 0.01][0]
p3 = (f01["dev_answerable"] == base["dev_answerable"]
      and f01["test_entry_1"] == base["test_entry_1"]
      and f01["conflict_hits"] == 2)
print("P3 a 0.01 floor is free            : %s (DEV %d/17, entry@1 %d/12, "
      "conflict %d/2, TEST silent %d -> %d)"
      % ("HELD" if p3 else "REFUTED", f01["dev_answerable"],
         f01["test_entry_1"], f01["conflict_hits"],
         base["test_silent"], f01["test_silent"]))

res = os.path.join(HERE, "results")
os.makedirs(res, exist_ok=True)
out = os.path.join(res, "ask_ranking.json")
json.dump({"baseline": base, "heading_damping": p1_rows,
           "pivoted_normalisation": p2_rows, "grid": grid,
           "floor": p3_rows}, io.open(out, "w", encoding="utf-8"), indent=1)
print("saved -> %s" % out)
