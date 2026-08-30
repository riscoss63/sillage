"""Can the memory tell when it does NOT know, from what it contributed?

On the Vernouil report the attribution separated cleanly: questions the
document answers moved 4-11 tokens of 12, questions it does not answer
moved 0-2. That is 12 points on the one document that suggested the
rule, so it is a hypothesis, not a result. This tests it on a corpus
that has set nothing.

The rule under test, fixed in advance from that document:

    answer if the memory moved >= 3 tokens of the first 12, else say
    the memory did not reach it.

The confound this has to survive: every answerable prompt there was a
VERBATIM prefix of the document, and every unanswerable one was an
invented sentence. `moved` might therefore be measuring surface overlap
rather than knowledge -- in which case the rule would go silent on any
genuine question a person rephrases. So the answerable set is split in
two, and the reworded half is the one that decides whether the signal
is worth anything.

Registered BEFORE the run:

  Y1  FALSE CONFIDENCE is what matters: on the UNANSWERABLE set the
      rule abstains at least 7 times out of 8.
      FALSIFIED below 6/8.
  Y2  On the VERBATIM answerable set the rule answers at least 7/8,
      and every answer it gives is correct.
      FALSIFIED if it answers and is wrong even once.
  Y3  On the REWORDED answerable set, the rule may abstain often --
      the memory is known to reach rephrased cues about 1 time in 10 --
      and that is acceptable. What is NOT acceptable is answering
      wrongly with confidence.
      FALSIFIED if any reworded question is answered (moved >= 3) with
      a wrong answer.
  Y4  The separation replicates: min(moved) over answered-correctly
      questions > max(moved) over unanswerable ones.
      FALSIFIED if the ranges overlap.

Run:  python behav/probe_abstain_gen.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sillage.runtime import Sillage          # noqa: E402

THRESHOLD = 3        # fixed from the Vernouil document, before this run
# 30, not the 12 of the first pass: at 12 a correct answer to "Qui a redige
# ce compte rendu ?" was cut off mid-sentence ("Le compte rendu a ete redige
# par le techn|") and scored as a confident error. It is not -- at 30 it
# reaches "monsieur Ovide Trenchard". A window that truncates the answer
# measures the window, not the memory.
N = 30

DOC = """Compte rendu de visite du rucher de Peyrelonge

La visite de printemps du rucher de Peyrelonge s'est deroulee le 11 avril
2026, par temps couvert et douze degres, en presence du referent sanitaire
du groupement.

La colonie de la ruche numero 7 occupe huit cadres de couvain sur les dix
que compte le corps. La reine, marquee en vert, a ete vue en ponte sur le
quatrieme cadre. Le taux d'infestation par varroa mesure au lange est de
2,4 acariens par jour, contre un seuil d'alerte fixe a 5 acariens par jour
pour la saison.

La ruche numero 12 est bourdonneuse depuis au moins trois semaines. Elle a
ete reunie a la ruche numero 9 par la methode du journal, apres retrait de
la fausse reine. Le poids de la ruche numero 9 apres reunion s'etablit a
trente-huit kilogrammes.

Les hausses ont ete posees sur cinq colonies. La miellee de colza est
estimee a quatorze kilogrammes par ruche sur la parcelle du bas, et la
floraison devrait durer encore dix jours.

Un nourrissement speculatif au sirop cinquante-cinquante a ete distribue
aux trois colonies les plus faibles, a raison d'un litre par colonie et par
semaine. Le compte rendu a ete redige par le technicien du groupement,
monsieur Ovide Trenchard, carte apicole numero 6318.
"""

# verbatim prefixes of the reflowed document
VERBATIM = [
    ("Le taux d'infestation par varroa mesure au lange est de", "2,4"),
    ("Le poids de la ruche numero 9 apres reunion s'etablit a", "trente-huit"),
    ("La miellee de colza est estimee a", "quatorze"),
    ("La colonie de la ruche numero 7 occupe", "huit"),
    ("Le compte rendu a ete redige par le technicien du groupement, monsieur",
     "Ovide"),
    ("La reine, marquee en vert, a ete vue en ponte sur le", "quatrieme"),
    ("Un nourrissement speculatif au sirop cinquante-cinquante a ete "
     "distribue aux", "trois"),
    ("La visite de printemps du rucher de Peyrelonge s'est deroulee le",
     "11 avril"),
]

# the same facts, asked the way a person would ask them
REWORDED = [
    ("Combien d'acariens varroa par jour ont ete comptes au lange ?", "2,4"),
    ("Quel est le poids de la ruche 9 une fois la reunion faite ?",
     "trente-huit"),
    ("Combien de kilos de colza par ruche sont attendus ?", "quatorze"),
    ("Sur combien de cadres s'etend le couvain de la ruche 7 ?", "huit"),
    ("Qui a redige ce compte rendu ?", "Ovide"),
    ("Sur quel cadre la reine a-t-elle ete apercue ?", "quatrieme"),
    ("Combien de colonies ont recu du sirop ?", "trois"),
    ("A quelle date la visite de printemps a-t-elle eu lieu ?", "11 avril"),
]

# nothing in the document answers these
UNANSWERABLE = [
    "Le rucher de Peyrelonge compte au total",
    "Le prix du kilo de miel de colza a la cooperative est de",
    "La prochaine visite du rucher de Peyrelonge aura lieu le",
    "Le modele des ruches utilisees au rucher de Peyrelonge est",
    "La production totale du rucher l'an dernier a atteint",
    "Le nom de la commune ou se trouve le rucher est",
    "L'age de la reine de la ruche numero 7 est de",
    "Le traitement anti-varroa retenu pour l'automne sera",
]


def run(s, items, kind):
    rows = []
    for item in items:
        prompt, want = item if isinstance(item, tuple) else (item, None)
        txt = s.complete(prompt, n=N, temp=0.0)
        at = s.attribution() or {}
        moved = at.get("moved") or 0
        answers = moved >= THRESHOLD
        correct = (want is not None and want.lower() in txt.lower())
        rows.append({"kind": kind, "prompt": prompt[-46:], "want": want,
                     "got": txt.strip()[:46], "moved": moved,
                     "answers": answers, "correct": correct})
        flag = ("ANSWER" if answers else "abstain")
        mark = "OK " if correct else (".  " if want else "   ")
        print(f"  [{kind:<9}] moved {moved:>2}/{N} {flag:<7} {mark} "
              f"{txt.strip()[:40]!r}", flush=True)
    return rows


def main():
    tmp = tempfile.mkdtemp(prefix="abstain_")
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        text = Sillage.reflow(DOC)
        for _ in range(2):
            rec = s.read_text(text)
        print(f"state: {rec['tokens']} tokens, {len(s.mem.cold)} grams, "
              f"ppl {rec['ppl_frozen']} -> {rec['ppl_with_memory']}\n",
              flush=True)
        rows = (run(s, VERBATIM, "verbatim")
                + run(s, REWORDED, "reworded")
                + run(s, UNANSWERABLE, "no-answer"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    def sel(k):
        return [r for r in rows if r["kind"] == k]

    ver, rew, non = sel("verbatim"), sel("reworded"), sel("no-answer")
    lied = [r for r in rew + ver if r["answers"] and not r["correct"]]
    answered_ok = [r for r in ver + rew if r["answers"] and r["correct"]]
    verdict = {
        "threshold": THRESHOLD,
        "Y1": {"abstained_on_unanswerable":
               sum(1 for r in non if not r["answers"]), "of": len(non),
               "holds": sum(1 for r in non if not r["answers"]) >= 7},
        "Y2": {"answered": sum(1 for r in ver if r["answers"]),
               "of": len(ver),
               "wrong_answers": [r["want"] for r in ver
                                 if r["answers"] and not r["correct"]],
               "holds": sum(1 for r in ver if r["answers"]) >= 7
               and not [r for r in ver if r["answers"] and not r["correct"]]},
        "Y3": {"reworded_answered": sum(1 for r in rew if r["answers"]),
               "reworded_correct": sum(1 for r in rew if r["correct"]),
               "confident_and_wrong": [r["want"] for r in rew
                                       if r["answers"] and not r["correct"]],
               "holds": not [r for r in rew
                             if r["answers"] and not r["correct"]]},
        "Y4": {"min_moved_when_right":
               min((r["moved"] for r in answered_ok), default=None),
               "max_moved_unanswerable": max(r["moved"] for r in non),
               "holds": bool(answered_ok) and
               min(r["moved"] for r in answered_ok)
               > max(r["moved"] for r in non)},
        "false_confidence_total": len(lied)}
    print("\n" + json.dumps(verdict, indent=1, ensure_ascii=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "abstain_gen.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "verdict": verdict}, fh, indent=1,
                  ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
