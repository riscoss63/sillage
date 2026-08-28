"""Kaggle kernel: paper-6 behavioral laws at capacity (Qwen3-4B).

Self-contained: pip installs sillage, everything else is synthetic (the
invented-fact dossier of the behavioral suite -- no state, no document,
no dataset attached). Two arms:

  N  native   Qwen3-4B (fp16) reads dossier v1 then v2 x2 itself;
              recall / paraphrase / locality witness (no write) /
              conflicts, then the trust probe: auto-fitted readout vs
              the family settings (40, 0.85, q50).
  T  transfer Qwen3-0.6B (fp32) builds v1+v2x2; then 0.6B, 1.7B (fp32)
              and 4B (fp16) serve THE SAME state -- the capacity axis
              at constant storage, published vs family readout.

Predictions were registered in the lab journal BEFORE this run and are
echoed in the output JSON. Declared caps: no interference arm at 4B;
the transfer state has no interference read (unlike the local
behav_qwen state -- within-kernel comparisons only); 4B forwards in
float16 (16 GB fp32 does not fit a T4).
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sillage", "--no-deps"], check=True)

import gc                                                      # noqa: E402
import json                                                    # noqa: E402
import os                                                      # noqa: E402
import shutil                                                  # noqa: E402
import time                                                    # noqa: E402

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

# ---- GPU self-diagnosis (the P100 trap) -- same pattern as paper 5 ------
if not torch.cuda.is_available():
    sys.exit("Pas de GPU alloue : Settings -> Accelerator -> GPU T4 x2.")
GPU = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {GPU} (sm_{cap[0]}{cap[1]}) | torch {torch.__version__}",
      flush=True)
if cap[0] < 7:
    print("\n" + "!" * 70)
    print(f"{GPU} (sm_{cap[0]}{cap[1]}) n'est PAS supporte par le torch "
          f"de l'image Kaggle (minimum sm_70).")
    print("Dans l'interface du kernel : Settings -> Accelerator -> "
          "GPU T4 x2, puis 'Save & Run All'. Rien d'autre a changer.")
    print("!" * 70, flush=True)
    sys.exit(0)

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

import sillage as _pkg                                         # noqa: E402
from sillage import Sillage                                    # noqa: E402

print("sillage", _pkg.__version__, flush=True)

WORK = "/kaggle/working"

PREDICTIONS = {
    "registered": "2026-08-27, avant le run (NOTES_AXE3.md)",
    "P0": "controles: memoire vide -> rappel = paraphrase = 0% aux deux "
          "capacites (sinon tout est invalide)",
    "P1": "4B natif: rappel v1 >= 80% (readout auto-ajuste), >= 93% "
          "(reglages famille)",
    "P2": "transfert, readout PUBLIE: conflit nouvelle <= 20% aux TROIS "
          "capacites (la capacite n'achete pas la conversion); reglages "
          "famille: nouvelle >= 90% aux trois. FALSIFICATION: 4B publie "
          ">= 50%",
    "P3": "paraphrase ~0% partout (frontiere de surface tokens)",
    "P4": "temoin 4B natif <= +3%",
    "P5": "stables servis >= 85% publie aux trois (ne chute pas en "
          "montant), >= 90% famille, croissance faible",
    "caps": "pas d'interference a 4B; etat transfert sans interference "
            "(comparaisons intra-kernel); 4B en float16",
}
print(json.dumps(PREDICTIONS, indent=1, ensure_ascii=False), flush=True)

# ---- constants copied verbatim from behav/behavioral.py -----------------
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
    hits = 0
    for e, v in facts:
        out = s.complete(prefix_tmpl.format(e=e), n=n)
        hits += v.split()[0] in out
    return hits / len(facts)


def nll_nowrite(s, text):
    """Teacher-forced (base_ppl, memory_ppl) WITHOUT any write."""
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


# ---- kernel-side helpers ------------------------------------------------
CALIBRATED = (40.0, 0.85, 0.5)          # the family settings of paper 5

facts = list(zip(ENTS[:30], VALS[:30]))
changed = [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]
old_facts = [(e, dict(facts)[e]) for e, _ in changed]
stable = [(e, v) for e, v in facts if e not in dict(changed)]
doc_v1 = build_doc(facts, seed=0)
doc_v2 = build_doc([(e, dict(changed).get(e, v)) for e, v in facts],
                   seed=50)


def hand(s, name, dtype):
    """Load the frozen model ourselves (fp16 when it must fit) and hand
    it to Sillage -- the 'model handed from outside' path of runtime."""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    m.to("cuda")
    m.eval()
    s._tok, s._model = tok, m
    print(f"  loaded {name} ({dtype}) in {time.time()-t0:.0f}s | "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    return s


def free(s):
    if s._model is not None:
        s._model = None
        s._tok = None
    gc.collect()
    torch.cuda.empty_cache()


def conflict(s, n=8):
    new = probe(s, changed, A_PREFIX, n)
    old = probe(s, old_facts, A_PREFIX, n)
    return {"new": new, "old": old, "neither": round(1 - new - old, 4)}


def readout_of(mem):
    return {"beta_G": float(mem.beta_G), "lam_G": float(mem.lam_G),
            "thr_qG": (None if mem.thr_qG is None else float(mem.thr_qG))}


R = {"predictions": PREDICTIONS,
     "config": {"gpu": GPU, "torch": torch.__version__,
                "sillage": _pkg.__version__,
                "dtype": {"0.6B": "float32", "1.7B": "float32",
                          "4B": "float16"},
                "facts": 30, "n": 8}}

# ======================= ARM N : native Qwen3-4B =========================
print("\n" + "=" * 70 + "\nARM N : Qwen3-4B natif (fp16)", flush=True)
RN = {}
ST_N = os.path.join(WORK, "state_4b")
shutil.rmtree(ST_N, ignore_errors=True)
sN = Sillage(model="Qwen/Qwen3-4B", state=ST_N, quiet=True)
hand(sN, "Qwen/Qwen3-4B", torch.float16)

print("== 0. controles avant lecture (memoire vide) ==", flush=True)
RN["recall_base"] = probe(sN, facts, A_PREFIX, 8)
RN["para_base"] = probe(sN, facts, B_PREFIX, 8)
b0, m0 = nll_nowrite(sN, WITNESS)
RN["witness_before"] = {"base": b0, "mem": m0}
print(f"  rappel base {RN['recall_base']:.0%} | paraphrase base "
      f"{RN['para_base']:.0%} | temoin PPL {b0:.2f} (mem {m0:.2f})",
      flush=True)

print("== 1-2. lecture v1, rappel + paraphrase ==", flush=True)
t0 = time.time()
sN.read_text(doc_v1, "dossier_v1")
sN.save()
print(f"  lu en {(time.time()-t0)/60:.1f} min", flush=True)
RN["recall_v1"] = probe(sN, facts, A_PREFIX, 8)
RN["para_v1"] = probe(sN, facts, B_PREFIX, 8)
RN["readout_after_v1"] = readout_of(sN.mem)
print(f"  rappel {RN['recall_v1']:.0%} | paraphrase {RN['para_v1']:.0%} | "
      f"readout {RN['readout_after_v1']}", flush=True)

print("== 4. localite (temoin, sans ecrire) ==", flush=True)
b1, m1 = nll_nowrite(sN, WITNESS)
RN["witness_after"] = {"base": b1, "mem": m1}
print(f"  temoin PPL mem {m0:.2f} -> {m1:.2f} "
      f"(delta {100*(m1-m0)/m0:+.1f}%)", flush=True)

print("== 5. conflits : v2 change 10 valeurs ==", flush=True)
sN.read_text(doc_v2, "dossier_v2")
sN.save()
RN["conflict_after_1"] = conflict(sN)
print(f"  apres v2 x1 : {RN['conflict_after_1']}", flush=True)
sN.read_text(doc_v2, "dossier_v2_bis")
sN.save()
RN["conflict_after_2"] = conflict(sN)
RN["readout_fitted"] = readout_of(sN.mem)
print(f"  apres v2 x2 : {RN['conflict_after_2']} | readout "
      f"{RN['readout_fitted']}", flush=True)

print("== 6. probe de confiance : reglages famille ==", flush=True)
sN.mem.beta_G, sN.mem.lam_G, sN.mem.thr_qG = CALIBRATED
RN["conflict_family"] = conflict(sN)
RN["stable_recall_family"] = probe(sN, stable, A_PREFIX, 8)
RN["recall_family"] = probe(sN, facts, A_PREFIX, 8)
RN["para_family"] = probe(sN, facts, B_PREFIX, 8)
print(f"  famille : conflit {RN['conflict_family']} | stables "
      f"{RN['stable_recall_family']:.0%} | rappel {RN['recall_family']:.0%}"
      f" | paraphrase {RN['para_family']:.0%}", flush=True)

R["native_4b"] = RN
free(sN)
del sN
json.dump(R, open(os.path.join(WORK, "behav_4b.json"), "w"), indent=2)

# ================ ARM T : capacity axis on ONE 0.6B state ================
print("\n" + "=" * 70 + "\nARM T : etat 0.6B servi par 0.6B / 1.7B / 4B",
      flush=True)
RT = {"servers": {}}
ST_T = os.path.join(WORK, "state_transfer")
shutil.rmtree(ST_T, ignore_errors=True)
sR = Sillage(model="qwen", state=ST_T, quiet=True)
hand(sR, sR.mem.hub, torch.float32)
t0 = time.time()
sR.read_text(doc_v1, "dossier_v1")
sR.read_text(doc_v2, "dossier_v2")
sR.read_text(doc_v2, "dossier_v2_bis")
sR.save()
print(f"  etat construit par 0.6B en {(time.time()-t0)/60:.1f} min "
      f"({sR.mem.tokens} tokens, {len(sR.mem.cold)} grams)", flush=True)
RT["state"] = {"tokens": int(sR.mem.tokens),
               "cold_grams": len(sR.mem.cold)}
free(sR)
del sR

# The semantic tier and the adapter are functions of the READING model's
# hidden geometry (paper 5 already turns the adapter off under --target);
# a 4B hidden state has no meaning for a whitening built on 0.6B vectors.
# The capacity axis therefore serves M_G + cold only, at ALL THREE
# capacities -- identical stack, only the language model grows.
RT["declared"] = ("serveurs a pile reduite: semantique et adaptateur "
                  "coupes aux trois capacites (geometrie du lecteur)")
for label, target, dtype in [("0.6B", None, torch.float32),
                             ("1.7B", "Qwen/Qwen3-1.7B", torch.float32),
                             ("4B", "Qwen/Qwen3-4B", torch.float16)]:
    print(f"-- serveur {label}", flush=True)
    s = Sillage(model="qwen", state=ST_T, target=target,
                fastweights=False, quiet=True)
    s.mem.semantic = False
    hand(s, target or s.mem.hub, dtype)
    e = {"published": {"readout": readout_of(s.mem),
                       "conflict": conflict(s),
                       "stable": probe(s, stable, A_PREFIX, 8)}}
    print(f"   publie : {e['published']['conflict']} | stables "
          f"{e['published']['stable']:.0%}", flush=True)
    s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = CALIBRATED
    e["family"] = {"conflict": conflict(s),
                   "stable": probe(s, stable, A_PREFIX, 8)}
    print(f"   famille: {e['family']['conflict']} | stables "
          f"{e['family']['stable']:.0%}", flush=True)
    RT["servers"][label] = e
    free(s)
    del s

R["transfer"] = RT
json.dump(R, open(os.path.join(WORK, "behav_4b.json"), "w"), indent=2)

# keep the output artifact small: states are rebuildable and never shipped
shutil.rmtree(ST_N, ignore_errors=True)
shutil.rmtree(ST_T, ignore_errors=True)
print("\nDONE -> behav_4b.json", flush=True)
