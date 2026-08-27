"""Defense test, score decomposition and write-level instrumentation for
the adversarial section of paper 6. Self-contained: rebuilds the dose-9
collision state (GPT-2), then runs three stages matching the committed
JSONs and the trace quoted in the paper:

  defense  sqrt-cold counts (refuted) + s_G / cold decomposition at the
           attacked addresses (results/behav_adversarial_gpt2_defense.json)
  trace    amp_write instrumentation on one witness entity: per-write
           (gate g, prior amplitude a, increment) for the true value and
           the distractor -- the measurement that resolved the 13:1
           anomaly into the gate-as-defense finding

    python adversarial_probes.py [defense|trace|all]
"""

import json
import os
import shutil
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage                                    # noqa: E402
from sillage.core import COLD_MIN_COUNT, D_K, NGRAM            # noqa: E402
from behavioral import A_PREFIX, ENTS, VALS, build_doc         # noqa: E402
from adversarial import DISTRACTOR, collision_doc, triple_probe  # noqa

FACTS = list(zip(ENTS[:30], VALS[:30]))
DOSES = ((1, 500), (2, 600), (6, 700))


def build_dose9(state):
    shutil.rmtree(state, ignore_errors=True)
    s = Sillage(model="gpt2", state=state, quiet=True)
    s.read_text(build_doc(FACTS, seed=0), "dossier")
    for add, seed in DOSES:
        s.read_text(collision_doc(FACTS, add, seed), f"c{add}")
    s.save()
    return s


def stage_defense(res_dir, n=8):
    state = os.path.join(HERE, ".adv_probe_state")
    s = build_dose9(state)
    mem = s.mem
    t, d, _ = triple_probe(s, FACTS, n)
    print(f"controle dose 9 : vrai {t:.0%} | distr {d:.0%}")

    def sqrt_cold(self, tok_next=None):
        if len(self._hist) < NGRAM:
            return None
        gram = np.array(self._hist[-NGRAM:], dtype=np.int32).tobytes()
        slot = self.cold.get(gram)
        if slot is None or sum(slot[1].values()) < COLD_MIN_COUNT:
            return None
        root = {k: np.sqrt(c) for k, c in slot[1].items()}
        tot = sum(root.values())
        if tok_next is not None:
            return root.get(int(tok_next), 0.0) / tot
        return {k: v / tot for k, v in root.items()}

    mem.cold_lookup = types.MethodType(sqrt_cold, mem)
    t2, d2, nn2 = triple_probe(s, FACTS, n)
    mem.cold_lookup = types.MethodType(type(mem).cold_lookup, mem)
    print(f"defense sqrt-cold : vrai {t2:.0%} | distr {d2:.0%}")

    tok, _ = s.load_model()
    rows = []
    for e, v in FACTS:
        ids = tok.encode(A_PREFIX.format(e=e))
        mem.new_stream()
        for tkn in ids:
            mem.step_key(int(tkn))
        q = mem._graw / np.sqrt(D_K)
        _, sG = mem.scores(mem.M, q)
        v_id = tok.encode(" " + v.split()[0])[0]
        d_id = tok.encode(" " + DISTRACTOR.split()[0])[0]
        pc = mem.cold_lookup()
        rows.append({"e": e, "sG_true": float(sG[v_id]),
                     "sG_distr": float(sG[d_id]),
                     "cold_true": pc.get(v_id, 0) if pc else None,
                     "cold_distr": pc.get(d_id, 0) if pc else None})
    mt = float(np.mean([r["sG_true"] for r in rows]))
    md = float(np.mean([r["sG_distr"] for r in rows]))
    print(f"s_G moyens : vrai {mt:.3f} | distr {md:.3f} -- gradient "
          f"d'ordre dans rows (entite 1 attaquee fraiche, suivantes "
          f"blindees par le gate)")
    json.dump({"control": {"true": t, "distr": d},
               "sqrt_cold": {"true": t2, "distr": d2, "neither": nn2},
               "sG_mean": {"true": mt, "distr": md}, "rows": rows},
              open(os.path.join(res_dir,
                                "behav_adversarial_gpt2_defense.json"),
                   "w"), indent=2)
    shutil.rmtree(state, ignore_errors=True)


def stage_trace(res_dir):
    state = os.path.join(HERE, ".adv_trace_state")
    shutil.rmtree(state, ignore_errors=True)
    s = Sillage(model="gpt2", state=state, quiet=True)
    tok, _ = s.load_model()
    E = "Vorlagune"
    val_id = tok.encode(" " + dict(FACTS)[E].split()[0])[0]
    fur_id = tok.encode(" " + DISTRACTOR.split()[0])[0]
    probe_gram = tuple(tok.encode(A_PREFIX.format(e=E))[-NGRAM:])

    mem = s.mem
    log = []
    orig = type(mem).amp_write

    def spy(self, M, q, u, tok_next, g):
        if tok_next in (val_id, fur_id) and M is self.M:
            a = max(0.0, float(u @ self.V[tok_next]))
            log.append({"tok": "VAL" if tok_next == val_id else "FUR",
                        "g": round(float(g), 2), "a": round(a, 3),
                        "inc": round(float(np.sqrt(a * a + g) - a), 3),
                        "at_probe_gram":
                            tuple(self._hist[-NGRAM:]) == probe_gram})
        return orig(self, M, q, u, tok_next, g)

    mem.amp_write = types.MethodType(spy, mem)
    s.read_text(build_doc(FACTS, seed=0), "dossier")
    for add, seed in DOSES:
        s.read_text(collision_doc(FACTS, add, seed), f"c{add}")
    for line in log:
        if line["at_probe_gram"]:
            print("  ", line)
    json.dump(log, open(os.path.join(res_dir,
                                     "behav_adversarial_trace.json"), "w"),
              indent=2)
    shutil.rmtree(state, ignore_errors=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    if which in ("defense", "all"):
        stage_defense(res_dir)
    if which in ("trace", "all"):
        stage_trace(res_dir)


if __name__ == "__main__":
    main()
