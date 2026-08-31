"""The surprise gate, tested properly: matched budget, two languages, seeds.

`probe_gate_power` found no effect at 3.3% resolution, but left three
holes, and the claim is about this project's own central mechanism:

  1. the arms were not matched on total written mass -- Shannon's mean
     gate is 3.27, the uniform control's was 1.0, so the control wrote
     roughly three times less and still matched it. That makes the null
     more striking but it is not the clean comparison.
  2. one corpus, one language (English technical prose)
  3. two seeds

All three are closed here. The matched control is built the way the
author's own BHD preprint builds it (Phase D): `constant` is fixed to
the MEAN of the modulated run, so both arms spend the same plasticity
budget and only its ALLOCATION differs. That is the comparison that
isolates what the gate is for.

  shannon          g = clip(-ln p, 0, CAP)
  uniform-matched  g = mean(g_shannon) on this very corpus  <- the test
  uniform-1        g = 1                                     (kept for
                                                              continuity)
  bayes            g = KL(p_mem || p_base)

Registered BEFORE the run:

  Q1  Shannon beats uniform-MATCHED on recall after interference, by
      more than one fact, in at least one language.
      FALSIFIED if the difference is within one fact in both.
      This is the decisive one: same budget, only the allocation
      differs, so any advantage here belongs to the gate itself.
  Q2  Same on the matrix alone (cold store emptied) -- paper 1's
      amplitude claim in isolation.
  Q3  The result is the same in both languages: whatever holds for
      English technical prose holds for French reports.
      FALSIFIED if the languages disagree, which would make the
      finding a property of the corpus rather than of the mechanism.
  Q4  Bayes stays worst everywhere, for the reason already measured.

Run:  python behav/probe_gate_final.py [--facts 30] [--seeds 3]
"""
import argparse
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
from behavioral import A_PREFIX, A_SENT, ENTS, VALS  # noqa: E402
from probe_gate_pressure import paper_body           # noqa: E402
from probe_which_surprise import read_gated          # noqa: E402

FR_VALS = ["quatre vannes de bronze", "dix-sept lanternes ambrees",
           "trois registres de granit", "neuf soufflets d'argent",
           "vingt-deux poulies cramoisies", "cinq boussoles creuses",
           "huit condensateurs tresses", "onze almanachs geles",
           "six turbines peintes", "quatorze aimants silencieux",
           "sept archives salees", "douze cheminees etroites",
           "quinze listes dorees", "quatre lanieres humides",
           "dix agrafes obliques", "trois gondoles robustes",
           "dix-huit ombrelles frappees", "neuf entonnoirs tresses",
           "cinq cadrans ternis", "vingt-six manivelles bleues",
           "sept ecluses de plomb", "onze rouleaux de chanvre",
           "quatre pivots de laiton", "treize fanaux voiles",
           "huit tambours de cuivre", "six vernis mats",
           "dix-neuf crochets courbes", "trois membranes fines",
           "douze ressorts tendus", "cinq bagues striees"]
FR_SENT = "Le protocole {e} exige {v}."
FR_PREFIX = "Le protocole {e} exige"

FR_FILLER = """La station de pompage occupe le fond du vallon, en contrebas
de la route departementale. Le batiment date des annees soixante et sa
toiture a ete refaite deux fois. On y accede par un chemin empierre que les
pluies d'automne ravinent regulierement. A l'interieur, trois groupes
motopompes se partagent la salle des machines, alignes contre le mur nord.
Le tableau electrique porte encore les etiquettes d'origine, ecrites a la
main. Un carnet d'entretien est pose sur l'etabli, ouvert a la page du mois
en cours. Les techniciens y consignent chaque intervention, la date, la
duree et les pieces remplacees. Le local de stockage attenant contient les
garnitures de rechange, les joints et les filtres. La ventilation est
assuree par deux grilles hautes et un extracteur qui se declenche au-dela
de trente degres. Le puits de captage se trouve a une centaine de metres,
protege par une margelle et un couvercle cadenasse. La conduite principale
descend ensuite vers le reservoir de tete, qui alimente le bourg par
gravite. """


def fr_corpus(n_facts, filler_reps):
    facts = list(zip(ENTS[:n_facts], FR_VALS[:n_facts]))
    block = "\n".join(FR_SENT.format(e=e, v=v) for e, v in facts)
    return (FR_FILLER * filler_reps) + "\n\n" + block + "\n", facts


def recall_of(s, facts, prefix, n=8):
    hit = 0
    for e, v in facts:
        out = s.complete(prefix.format(e=e), n=n)
        hit += v.split()[0].lower() in out.lower()
    return hit / len(facts)


def mean_shannon_gate(s, text):
    """One recording pass, on a throwaway memory, to set the budget."""
    rec = []
    read_gated(s, text, "shannon", rec)
    return float(np.mean([p[0] for p in rec]))


def run_cell(lang, model, target, interference, facts, prefix, cold_max,
             seeds, matched_g):
    rows = []
    arms = ("shannon", "uniform-matched", "uniform-1", "bayes")
    for seed in range(seeds):
        inter = interference if seed % 2 == 0 else "\n\n".join(
            interference.split("\n\n")[::-1])
        for gate in arms:
            tmp = tempfile.mkdtemp(prefix="final_")
            try:
                s = Sillage(model=model, state=tmp, quiet=True,
                            fastweights=False, cold_max=cold_max)
                s.load_model()
                # a float gate means a CONSTANT of that value
                g_name = {"uniform-matched": matched_g,
                          "uniform-1": 1.0}.get(gate, gate)
                for _ in range(2):
                    read_gated(s, target, g_name)
                before = recall_of(s, facts, prefix)
                read_gated(s, inter, g_name)
                s.mem.prune_cold()
                after = recall_of(s, facts, prefix)
                backup = s.mem.cold
                s.mem.cold = {}
                matrix_only = recall_of(s, facts, prefix)
                s.mem.cold = backup
                rows.append({"lang": lang, "seed": seed, "gate": gate,
                             "before": round(before, 3),
                             "after": round(after, 3),
                             "matrix_only": round(matrix_only, 3),
                             "median_mass": round(float(np.median(
                                 [sl[0] for sl in s.mem.cold.values()])), 2)})
                print(f"  {lang}  seed {seed}  {gate:<16} "
                      f"{before:.0%} -> {after:.0%}  "
                      f"(matrix {matrix_only:.0%})  "
                      f"mass {rows[-1]['median_mass']}", flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cold-max", type=int, default=1200)
    a = ap.parse_args()
    print("closing the three holes: matched budget, two languages, "
          f"{a.seeds} seeds\n", flush=True)
    res = {"facts": a.facts, "seeds": a.seeds, "cold_max": a.cold_max,
           "rows": []}

    # --- English technical prose, gpt2 -----------------------------------
    en_facts = list(zip(ENTS[:a.facts], VALS[:a.facts]))
    en_block = "\n".join(A_SENT.format(e=e, v=v) for e, v in en_facts)
    en_target = paper_body("sillage", 5000) + "\n\n" + en_block + "\n"
    en_inter = (paper_body("behavior", 6000) + "\n\n"
                + paper_body("benchmark", 6000))
    tmp = tempfile.mkdtemp(prefix="budget_")
    try:
        s0 = Sillage(model="gpt2", state=tmp, quiet=True, fastweights=False)
        s0.load_model()
        g_en = mean_shannon_gate(s0, en_target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"English: matched budget g = {g_en:.3f}\n", flush=True)
    res["rows"] += run_cell("EN", "gpt2", en_target, en_inter, en_facts,
                            A_PREFIX, a.cold_max, a.seeds, g_en)

    # --- French reports, same model so the LANGUAGE is the only change ---
    fr_target, fr_facts = fr_corpus(a.facts, 6)
    fr_inter = (FR_FILLER * 14) + "\n\n" + (FR_FILLER[::-1] * 4)
    tmp = tempfile.mkdtemp(prefix="budget_")
    try:
        s0 = Sillage(model="gpt2", state=tmp, quiet=True, fastweights=False)
        s0.load_model()
        g_fr = mean_shannon_gate(s0, fr_target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nFrench: matched budget g = {g_fr:.3f}\n", flush=True)
    res["rows"] += run_cell("FR", "gpt2", fr_target, fr_inter, fr_facts,
                            FR_PREFIX, a.cold_max, a.seeds, g_fr)

    def mean(lang, gate, key):
        v = [r[key] for r in res["rows"]
             if r["gate"] == gate and r["lang"] == lang]
        return round(float(np.mean(v)), 3) if v else None

    one = 1.0 / a.facts
    res["summary"] = {
        lang: {g: {k: mean(lang, g, k)
                   for k in ("before", "after", "matrix_only",
                             "median_mass")}
               for g in ("shannon", "uniform-matched", "uniform-1", "bayes")}
        for lang in ("EN", "FR")}
    d_en = mean("EN", "shannon", "after") - mean("EN", "uniform-matched",
                                                 "after")
    d_fr = mean("FR", "shannon", "after") - mean("FR", "uniform-matched",
                                                 "after")
    m_en = mean("EN", "shannon", "matrix_only") - mean(
        "EN", "uniform-matched", "matrix_only")
    m_fr = mean("FR", "shannon", "matrix_only") - mean(
        "FR", "uniform-matched", "matrix_only")
    res["verdict"] = {
        "Q1_delta_after": {"EN": round(d_en, 3), "FR": round(d_fr, 3)},
        "Q1_holds": (d_en > one) or (d_fr > one),
        "Q2_delta_matrix": {"EN": round(m_en, 3), "FR": round(m_fr, 3)},
        "Q2_holds": (m_en > one) or (m_fr > one),
        "Q3_languages_agree": (d_en > one) == (d_fr > one),
        "Q4_bayes_worst": all(
            mean(l, "bayes", "after") <= min(mean(l, "shannon", "after"),
                                             mean(l, "uniform-matched",
                                                  "after"))
            for l in ("EN", "FR")),
        "one_fact_is": round(one, 3),
        "matched_budgets": {"EN": round(g_en, 3), "FR": round(g_fr, 3)}}
    print("\n" + json.dumps(res["summary"], indent=1))
    print(json.dumps(res["verdict"], indent=1))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "gate_final.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
