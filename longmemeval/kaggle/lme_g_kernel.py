"""Kaggle kernel: LongMemEval-S, arm G -- the generative voices.

43 questions (40 stratified by type + 3 _abs, seed 7), Qwen3-0.6B.
Stack: M_G + cold + semantic tier, adapter OFF (style tier; the
target-serving precedent). Ingestion via fast_ingest gate=torch
(local tolerance test passed: identical cold admissions, ~1e-6 trace
drift). Three voices per question, greedy n=24, deterministic scoring
(normalized answer containment):

  a  memory-only      "Question: ...\nAnswer:" on the state of the
                      ~50 ingested sessions (~121k tokens)
  b  context+memory   top-3 index passages + question, same state
  c  context-only     same prompt as b, EMPTY state (bare model)

Predictions registered in the lab journal before the run, echoed in
the output JSON. Results written incrementally after every question.
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sillage", "--no-deps"], check=True)

import gc                                                      # noqa: E402
import glob                                                    # noqa: E402
import json                                                    # noqa: E402
import os                                                      # noqa: E402
import random                                                  # noqa: E402
import re                                                      # noqa: E402
import shutil                                                  # noqa: E402
import time                                                    # noqa: E402

import torch                                                   # noqa: E402

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

# ---- locate the kit (dataset mount layouts vary) ------------------------
DATA = CODE = None
for attempt in range(12):
    hits = glob.glob("/kaggle/input/**/longmemeval_s", recursive=True)
    if hits:
        DATA = hits[0]
        CODE = os.path.dirname(hits[0])
        break
    print(f"kit pas encore visible (essai {attempt+1}/12)", flush=True)
    time.sleep(30)
if DATA is None:
    sys.exit("longmemeval_s introuvable : attacher le dataset "
             "sillage-lme-kit au kernel.")
sys.path.insert(0, CODE)
print("kit:", CODE, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402

import sillage as _pkg                                         # noqa: E402
from sillage import Sillage                                    # noqa: E402
from sillage.index import Index                                # noqa: E402
from fast_ingest import fast_ingest_blocked                    # noqa: E402

print("sillage", _pkg.__version__, flush=True)
WORK = "/kaggle/working"

PREDICTIONS = {
    "registered": "2026-08-27, avant le run (NOTES_AXE3.md)",
    "P-G1": "voix (a) memoire seule <= 10% global (double mur "
            "confiance + formulation a 121k tokens d'etat)",
    "P-G2": "voix (b) contexte+memoire = voix (c) contexte seul a +-5 "
            "points (redondance en fenetre tarifee ~0). FALSIFICATION: "
            "(b) < (c) - 10 points",
    "P-G3": "(b) et (c) >= 4x la voix (a) -- la formulation se fait en "
            "fenetre",
    "P-G4": "ingestion >= 100 tok/s (contre 7 tok/s en lecture pleine)",
    "caps": "adaptateur coupe (tier de style); gate=torch (tolerance "
            "validee localement); 43 questions graine 7; scoring = "
            "containment normalise (preference hors-proxy)",
}
print(json.dumps(PREDICTIONS, indent=1, ensure_ascii=False), flush=True)

# ---- sample: 40 stratified + 3 _abs, seed 7 -----------------------------
QUOTA = {"temporal-reasoning": 10, "multi-session": 10,
         "knowledge-update": 6, "single-session-user": 6,
         "single-session-assistant": 5, "single-session-preference": 3}

d = json.load(open(DATA, encoding="utf-8"))
rng = random.Random(7)
core = [q for q in d if not str(q["question_id"]).endswith("_abs")]
absq = [q for q in d if str(q["question_id"]).endswith("_abs")]
sample = []
for t, k in sorted(QUOTA.items()):
    pool = sorted((q for q in core if q["question_type"] == t),
                  key=lambda q: str(q["question_id"]))
    sample += rng.sample(pool, k)
sample += rng.sample(sorted(absq, key=lambda q: str(q["question_id"])), 3)
print(f"{len(sample)} questions echantillonnees", flush=True)


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def session_text(turns):
    return "\n\n".join(f"{t['role']}: {t['content']}" for t in turns)


def hit(answer, out):
    return norm(answer) in norm(out)


# one shared frozen model for every state (handed from outside)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.float32).to("cuda").eval()


def assistant(state):
    s = Sillage(model="qwen", state=state, fastweights=False, quiet=True)
    s._tok, s._model = tok, model
    s.device = "cuda"
    return s


R = {"predictions": PREDICTIONS,
     "config": {"gpu": GPU, "torch": torch.__version__,
                "sillage": _pkg.__version__, "n_gen": 24,
                "stack": "M_G + cold + semantique, adaptateur OFF",
                "gate": "torch", "ingest": "blocked-64", "seed": 7},
     "rows": []}
OUT = os.path.join(WORK, "lme_arm_g.json")

T_START = time.time()
for qi, q in enumerate(sample):
    t0 = time.time()
    st = os.path.join(WORK, "st_q")
    shutil.rmtree(st, ignore_errors=True)
    s = assistant(st)
    ix = Index(None)
    n_tok = 0
    for sid, turns in zip(q["haystack_session_ids"],
                          q["haystack_sessions"]):
        text = session_text(turns)
        ix.add(text, str(sid))
        rec = fast_ingest_blocked(s, text, str(sid))
        n_tok += rec["tokens"]
    mins = (time.time() - t0) / 60
    rate = n_tok / max(1e-6, mins * 60)

    hits3 = ix.search(q["question"], k=3)
    ctx = "\n\n".join(h[1]["text"] for h in hits3)
    ans_ids = {str(a) for a in q["answer_session_ids"]}
    ev3 = any(h[1]["source"] in ans_ids for h in hits3)

    p_a = f"Question: {q['question']}\nAnswer:"
    p_bc = (f"{ctx}\n\nQuestion: {q['question']}\nAnswer:")
    out_a = s.complete(p_a, n=24)
    out_b = s.complete(p_bc, n=24)
    del s
    gc.collect()
    torch.cuda.empty_cache()

    st0 = os.path.join(WORK, "st_empty")
    shutil.rmtree(st0, ignore_errors=True)
    s0 = assistant(st0)
    out_c = s0.complete(p_bc, n=24)
    del s0
    gc.collect()

    row = {"id": q["question_id"], "type": q["question_type"],
           "abs": str(q["question_id"]).endswith("_abs"),
           "tokens": n_tok, "ingest_min": round(mins, 2),
           "tok_per_s": round(rate, 1), "evidence@3": ev3,
           "a_mem": hit(q["answer"], out_a),
           "b_ctx_mem": hit(q["answer"], out_b),
           "c_ctx_only": hit(q["answer"], out_c),
           "out_a": out_a[:80], "out_b": out_b[:80],
           "out_c": out_c[:80]}
    R["rows"].append(row)
    json.dump(R, open(OUT, "w"), indent=1)
    print(f"[{qi+1}/{len(sample)}] {q['question_type'][:18]:18s} "
          f"{n_tok:6d} tok @ {rate:5.0f} tok/s | a={row['a_mem']} "
          f"b={row['b_ctx_mem']} c={row['c_ctx_only']} ev3={ev3}",
          flush=True)
    # a clean sys.exit(0) PERSISTS /kaggle/working; the 12h timeout
    # kill does not -- never let a slow run ride to its death
    elapsed_h = (time.time() - T_START) / 3600
    projected_h = elapsed_h / (qi + 1) * len(sample)
    if projected_h > 9.5:
        print("", flush=True)
        print(f"ABANDON PROPRE : {elapsed_h:.1f} h pour {qi+1} "
              f"questions -> {projected_h:.1f} h projetees (> 9.5 h, "
              f"la session serait tuee sans sauvegarde). Les "
              f"{qi+1} lignes sont dans lme_arm_g.json.", flush=True)
        R["aborted"] = {"after": qi + 1,
                        "projected_hours": round(projected_h, 1)}
        json.dump(R, open(OUT, "w"), indent=1)
        sys.exit(0)

core_rows = [r for r in R["rows"] if not r["abs"]]


def agg(key):
    return sum(r[key] for r in core_rows) / max(1, len(core_rows))


R["overall"] = {"n_core": len(core_rows),
                "a_mem": agg("a_mem"), "b_ctx_mem": agg("b_ctx_mem"),
                "c_ctx_only": agg("c_ctx_only"),
                "evidence@3": agg("evidence@3"),
                "median_tok_per_s": sorted(
                    r["tok_per_s"] for r in R["rows"])[len(R["rows"])//2]}
json.dump(R, open(OUT, "w"), indent=1)
print("\n== global (hors _abs) ==")
for k in ("a_mem", "b_ctx_mem", "c_ctx_only", "evidence@3"):
    print(f"  {k:12s}: {R['overall'][k]:.1%}")
print(f"  ingest median: {R['overall']['median_tok_per_s']:.0f} tok/s")
shutil.rmtree(os.path.join(WORK, "st_q"), ignore_errors=True)
shutil.rmtree(os.path.join(WORK, "st_empty"), ignore_errors=True)
print("\nDONE -> lme_arm_g.json", flush=True)
