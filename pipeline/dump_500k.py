"""GPT-2 base pass over the two 500k-token streams (same windowed protocol
as dump_base.py). Expect ~35 min per stream on CPU."""


# --- repo bootstrap: run this script from anywhere ---
import os as _os
import sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "requirements.txt")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
for _sub in ("", "pipeline", "memory", "fastweights", "eval", "figures"):
    _p = _os.path.join(_d, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)
# --- end bootstrap ---

import json
import os
import time

import numpy as np
import torch

from dump_base import dump_domain

DATA, DUMPS = "data", "dumps"


def main():
    os.makedirs(DUMPS, exist_ok=True)
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(os.cpu_count() or 4)
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    report_path = os.path.join(DUMPS, "base_report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    for d in ["tolstoy", "bible"]:
        ids = np.load(os.path.join(DATA, f"{d}_ids.npy"))
        t0 = time.time()
        H, LP = dump_domain(model, ids)
        np.save(os.path.join(DUMPS, f"{d}_h.npy"), H)
        np.save(os.path.join(DUMPS, f"{d}_lp.npy"), LP)
        report[d] = {"tokens": int(len(ids)),
                     "base_ppl": float(np.exp(-LP.mean())),
                     "base_nll": float(-LP.mean()),
                     "minutes": round((time.time() - t0) / 60, 1)}
        print(f"{d}: N={len(ids)} base PPL={report[d]['base_ppl']:.2f} "
              f"({report[d]['minutes']} min)", flush=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
