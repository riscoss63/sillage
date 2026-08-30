"""Does paper 5's family readout make the memory answer on REAL prose?

behav_4b.json measured the dial on the synthetic protocol -- thirty
invented facts, canonical prefixes, a purpose-built conflict set -- and
found 10% conversion with the shipped readout against 100% with the
family settings, at 0.6B, 1.7B and 4B alike. That protocol is the
paper's own instrument. This probe asks the question the tool's users
actually ask: on an ordinary French document, read once, does the dial
turn a mostly-silent `complete` into one that answers -- and what does
it cost on a document the memory never read.

Registered BEFORE the run, with falsification thresholds:

  P1  Canonical recall rises. family >= published, and family >= 5/8.
      FALSIFIED if family < published.
  P2  The memory speaks far more: mean moved-token fraction at least
      doubles, published -> family.
      FALSIFIED if the increase is under 1.5x.
  P3  Locality is paid for. The witness document's perplexity rises
      more under family than under published.
      Recorded either way; a rise over +1.00 nat is a real cost that
      has to be reported next to the recall gain.
  P4  Invention on unanswerable questions is NOT fixed by the dial, and
      may get worse: family produces at least as many fabricated
      answers as published.
      FALSIFIED if family invents strictly less.
  P5  Attribution is informative: on at least one unanswerable question
      the memory moves ZERO tokens, showing the fabrication is the
      frozen model's and not a recall.
      FALSIFIED if the memory moves tokens everywhere.

Run:  python behav/probe_readout_dial.py [--model qwen] [--target HUB]
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

from sillage.runtime import Sillage           # noqa: E402
from sillage.cli import FAMILY_READOUT, apply_readout   # noqa: E402


def nll_nowrite(s, text):
    """Teacher-forced (base_ppl, memory_ppl) WITHOUT any write.

    Lifted verbatim from behav/kaggle_4b/behav4b_kernel.py so the
    locality witness here is the same instrument as the 4B run's.
    """
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

DOC = """Rapport d'intervention 2026-114 - Station de pompage de Vernouil

L'intervention a ete conduite le 14 juin 2026 par l'equipe de maintenance
sur la station de pompage de Vernouil, en presence du responsable
d'exploitation. Le motif de la visite etait une chute de debit signalee
par l'exploitant depuis le 2 juin.

Le groupe motopompe numero 3 presentait un jeu radial de 0,42 millimetre
au palier avant, contre 0,15 millimetre a la mise en service. La garniture
mecanique a ete remplacee par une reference CarbTec 88-R, et le couple de
serrage applique aux huit goujons de bride est de 62 newtons-metres.

La pression de refoulement mesuree apres intervention s'etablit a 4,7 bars
pour un debit de 118 metres cubes par heure, contre 3,1 bars et 74 metres
cubes par heure avant l'intervention. La temperature du palier arriere est
redescendue de 78 degres a 51 degres.

Le filtre amont a ete nettoye et son encrassement estime a 40 pour cent de
la section utile. Le compteur horaire du groupe numero 3 affiche 21 480
heures de fonctionnement.

Une anomalie residuelle est notee : le clapet anti-retour de la conduite
secondaire montre un leger battement a bas debit. L'equipe recommande son
remplacement, sans caractere d'urgence.

Le rapport a ete signe le 14 juin 2026 par le technicien responsable,
madame Brindas Kolvec, matricule 4471.
"""

WITNESS = """La cuisson du pain au levain demande une attention constante a la
temperature de la piece. Un levain jeune double de volume en quatre a six
heures a vingt-quatre degres, mais il lui faut le double de temps dans une
cuisine fraiche d'hiver. Le petrissage se fait en deux temps, avec un repos
d'une demi-heure entre les deux, ce qui laisse le gluten se detendre. La
farine complete boit davantage d'eau que la farine blanche, et il faut donc
augmenter l'hydratation d'environ cinq pour cent quand on en met un tiers
dans le melange. La cuisson demarre a four tres chaud, avec de la vapeur
pendant les vingt premieres minutes, puis se poursuit a chaleur moderee.
"""

# (prompt, the string that makes the answer correct)
ANSWERABLE = [
    ("Le couple de serrage applique aux huit goujons de bride est de",
     "62"),
    ("La pression de refoulement mesuree apres intervention s'etablit a",
     "4,7"),
    ("Le groupe motopompe numero 3 presentait un jeu radial de",
     "0,42"),
    ("La garniture mecanique a ete remplacee par une reference",
     "CarbTec"),
    ("Le compteur horaire du groupe numero 3 affiche",
     "21 480"),
    ("La temperature du palier arriere est redescendue de 78 degres a",
     "51"),
    ("Le rapport a ete signe le 14 juin 2026 par le technicien responsable, "
     "madame",
     "Brindas"),
    ("Le filtre amont a ete nettoye et son encrassement estime a",
     "40"),
]

# nothing in the document answers these
UNANSWERABLE = [
    "La prochaine visite de la station de Vernouil aura lieu le",
    "Le cout total de l'intervention sur la station de Vernouil s'eleve a",
    "Le fabricant du groupe motopompe numero 3 est",
    "La station de pompage de Vernouil alimente une population de",
]


def read_doc(s, text, passes=2):
    """Read the report, twice: the two-occurrence rule is the tool's own."""
    rec = None
    for _ in range(passes):
        rec = s.read_text(text)
    return rec


def set_readout(s, triple):
    """Restore an exact triple -- `apply_readout(s, 'published')` is a
    no-op by design (the CLI builds a fresh object), so a probe that
    switches arms in one process has to put the numbers back itself."""
    s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = triple


def run_arm(s, spec, label):
    set_readout(s, spec)
    out = {"readout": [float(s.mem.beta_G), float(s.mem.lam_G),
                       float(s.mem.thr_qG)], "answerable": [],
           "unanswerable": []}
    hit = 0
    for prompt, want in ANSWERABLE:
        txt = s.complete(prompt, n=12, temp=0.0)
        at = s.attribution() or {}
        ok = want.lower() in txt.lower()
        hit += ok
        out["answerable"].append(
            {"prompt": prompt[-42:], "want": want, "got": txt.strip()[:60],
             "ok": bool(ok), "moved": at.get("moved"),
             "tokens": at.get("tokens"), "tiers": at.get("tiers")})
        print(f"  [{label}] {'OK ' if ok else '   '} {want:<9} <- "
              f"{txt.strip()[:46]!r} (moved {at.get('moved')}/"
              f"{at.get('tokens')})", flush=True)
    out["recall"] = hit / len(ANSWERABLE)
    for prompt in UNANSWERABLE:
        txt = s.complete(prompt, n=12, temp=0.0)
        at = s.attribution() or {}
        out["unanswerable"].append(
            {"prompt": prompt[-42:], "got": txt.strip()[:60],
             "moved": at.get("moved"), "tokens": at.get("tokens"),
             "tiers": at.get("tiers")})
        print(f"  [{label}] ???            <- {txt.strip()[:46]!r} "
              f"(moved {at.get('moved')}/{at.get('tokens')})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--target", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="readout_dial_")
    res = {"predictions": __doc__.split("Registered BEFORE")[1]
           .split("Run:")[0].strip(),
           "model": a.model, "target": a.target}
    try:
        s = Sillage(model=a.model, state=tmp, target=a.target, quiet=True)
        print("reading the report twice ...", flush=True)
        rec = read_doc(s, DOC)
        res["state"] = {"tokens": rec["tokens"],
                        "cold_grams": len(s.mem.cold),
                        "ppl_frozen": rec["ppl_frozen"],
                        "ppl_with_memory": rec["ppl_with_memory"]}
        print(f"  state: {res['state']}", flush=True)

        pub = (float(s.mem.beta_G), float(s.mem.lam_G), float(s.mem.thr_qG))
        arms = ((pub, "published"), (FAMILY_READOUT, "family"))
        res["arms"] = {"published": list(pub), "family": list(FAMILY_READOUT)}

        # The recall arms run FIRST. `nll_nowrite` folds every state it
        # scores into the v1 tier's running centre `mu` WITHOUT the
        # matching writes to the tiers -- an imbalance only probe code
        # produces -- and measured that way it costs a fact (8/8 -> 7/8
        # on this document). Ordinary reading moves the centre and the
        # stored keys together and costs nothing (results/moredocs.json),
        # so the witness stays a valid locality measure; it just cannot
        # run before what it would perturb.
        for triple, label in arms:
            print(f"\n=== {label} readout {triple} ===", flush=True)
            res[label] = run_arm(s, triple, label)
            print(f"  recall {res[label]['recall']:.0%}", flush=True)

        for triple, label in arms:
            set_readout(s, triple)
            b, m = nll_nowrite(s, WITNESS)
            res.setdefault("witness", {})[label] = {"base": b, "mem": m}
            print(f"  witness {label}: base {b:.4f} mem {m:.4f} "
                  f"({m - b:+.4f})", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "readout_dial.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
