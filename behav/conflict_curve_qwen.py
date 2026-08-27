"""Extend the Qwen3 conflict curve to v2 x3 and x4 on the state left by
`behavioral.py --model qwen --state .behav_state_qwen`
(results/behav_qwen_curve.json)."""

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
from behavioral import ALT, A_PREFIX, ENTS, VALS, build_doc, probe  # noqa

STATE = os.path.join(HERE, ".behav_state_qwen")

facts = list(zip(ENTS[:30], VALS[:30]))
changed = [(e, ALT[i]) for i, (e, _v) in enumerate(facts[:10])]
doc_v2 = build_doc([(e, dict(changed).get(e, v)) for e, v in facts],
                   seed=50)

if not os.path.exists(STATE):
    raise SystemExit("state not found -- build it first:\n"
                     "  python behavioral.py --model qwen "
                     "--state .behav_state_qwen")
s = Sillage(model="qwen", state=STATE, quiet=True)
R = {}
for k in (3, 4):
    s.read_text(doc_v2, f"dossier_v2_x{k}")
    s.save()
    new, _ = probe(s, changed, A_PREFIX, 8)
    old, _ = probe(s, [(e, dict(facts)[e]) for e, _ in changed],
                   A_PREFIX, 8)
    R[f"x{k}"] = {"new": new, "old": old, "neither": 1 - new - old}
    print(f"apres v2 x{k} : nouvelle {new:.0%} | ancienne {old:.0%} | "
          f"confusion {1-new-old:.0%}", flush=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(R, open(os.path.join(HERE, "results",
                               "behav_qwen_curve.json"), "w"), indent=2)
print("saved -> results/behav_qwen_curve.json")
