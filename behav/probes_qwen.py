"""Read-only probes on the Qwen3 behavioral state (paper 6, sections on
trust, trade-offs and locality). Requires the state left by
`behavioral.py --model qwen --state .behav_state_qwen` extended by
`conflict_curve_qwen.py` (v2 read four times in total).

Stages, matching the committed JSONs:
  readout   the trust probe: same state, calibrated-trust settings
            (results/behav_qwen_readout_probe.json)
  tradeoff  paraphrase at high trust + witness PPL under both readouts
            (results/behav_qwen_tradeoff.json)
  locality  witness ablation: adapter / semantic tier
            (results/behav_qwen_locality_ablation.json)

    python probes_qwen.py [readout|tradeoff|locality|all]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage import Sillage                                    # noqa: E402
from behavioral import (ALT, A_PREFIX, B_PREFIX, ENTS, VALS,   # noqa
                        WITNESS, nll_nowrite, probe)

STATE = os.path.join(HERE, ".behav_state_qwen")
CALIBRATED = (40.0, 0.85, 0.5)      # the family settings of paper 5


def load():
    if not os.path.exists(STATE):
        raise SystemExit(
            "state not found -- build it first:\n"
            "  python behavioral.py --model qwen --state .behav_state_qwen\n"
            "  python conflict_curve_qwen.py")
    return Sillage(model="qwen", state=STATE, quiet=True)


def stage_readout(s, res_dir, n=8):
    facts = list(zip(ENTS[:30], VALS[:30]))
    changed = [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]
    old_facts = [(e, dict(facts)[e]) for e, _ in changed]
    new0, _ = probe(s, changed, A_PREFIX, n)
    s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = CALIBRATED
    new1, _ = probe(s, changed, A_PREFIX, n)
    old1, _ = probe(s, old_facts, A_PREFIX, n)
    stable = [(e, v) for e, v in facts if e not in dict(changed)]
    rec, _ = probe(s, stable, A_PREFIX, n)
    out = {"published": new0, "calibrated": {"new": new1, "old": old1},
           "stable_recall_calibrated": rec}
    print(f"readout: publie {new0:.0%} -> calibre nouvelle {new1:.0%} "
          f"(ancienne {old1:.0%}) | stables {rec:.0%}")
    json.dump(out, open(os.path.join(res_dir,
                                     "behav_qwen_readout_probe.json"), "w"),
              indent=2)


def stage_tradeoff(s, res_dir, n=8):
    facts = list(zip(ENTS[:30], VALS[:30]))
    b0, m0 = nll_nowrite(s, WITNESS)
    s.mem.beta_G, s.mem.lam_G, s.mem.thr_qG = CALIBRATED
    b1, m1 = nll_nowrite(s, WITNESS)
    para, _ = probe(s, facts, B_PREFIX, n)
    out = {"witness_published": {"base": b0, "mem": m0},
           "witness_calibrated": {"base": b1, "mem": m1},
           "paraphrase_calibrated": para}
    print(f"tradeoff: temoin {m0:.2f} / {m1:.2f} (publie/calibre) | "
          f"paraphrase a haute confiance {para:.0%}")
    json.dump(out, open(os.path.join(res_dir,
                                     "behav_qwen_tradeoff.json"), "w"),
              indent=2)


def stage_locality(s, res_dir):
    res = {}

    def run(tag):
        b, m = nll_nowrite(s, WITNESS)
        res[tag] = {"base": b, "mem": m, "delta_pct": 100 * (m - b) / b}
        print(f"  {tag:30s}: delta {100*(m-b)/b:+.2f}%")

    run("tout actif")
    s.mem.fastweights = False
    run("sans adaptateur")
    s.mem.semantic = False
    run("sans adaptateur ni semantique")
    s.mem.fastweights = True
    run("adaptateur seul (sans sem.)")
    json.dump(res, open(os.path.join(
        res_dir, "behav_qwen_locality_ablation.json"), "w"), indent=2)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res_dir = os.path.join(HERE, "results")
    os.makedirs(res_dir, exist_ok=True)
    if which in ("readout", "all"):
        stage_readout(load(), res_dir)
    if which in ("tradeoff", "all"):
        stage_tradeoff(load(), res_dir)
    if which in ("locality", "all"):
        stage_locality(load(), res_dir)


if __name__ == "__main__":
    main()
