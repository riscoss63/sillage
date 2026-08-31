"""Can the memory autocomplete on its own, with no model in the loop?

The cold store is an exact 4-gram -> successor table. Following its own
successors needs a tokenizer and a dict, and nothing else: no forward
pass, no torch, no GPU. `probe_lamc` already pulled " Brindas Kolvec"
out of it that way, as a side measurement.

If that holds, autocomplete is the product this project has been looking
for, because it inverts every weakness at once:

  * latency is a dict lookup rather than 500 ms of decoding
  * it CANNOT fabricate -- it only ever emits what it actually read
  * it is purely the memory, without the 0.6B model that drags
  * a wrong suggestion costs the typist nothing

Registered BEFORE the run:

  H1  Latency: a suggestion is produced in under 1 ms, excluding
      tokenisation. FALSIFIED above 5 ms.
  H2  Coverage: on prefixes taken from a document the memory has read,
      at least 60% of positions yield a suggestion of >= 2 tokens.
      FALSIFIED below 40% -- the feature would be silent too often to
      feel alive.
  H3  Correctness: every suggestion offered is a VERBATIM continuation
      of the document at that point. This is the whole claim, so any
      counter-example falsifies it.
  H4  Silence off-corpus: on prefixes from a document the memory has
      NOT read, coverage is under 10%. FALSIFIED above 25% -- it would
      be completing from noise.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sillage.core as core                          # noqa: E402
from sillage.runtime import Sillage                  # noqa: E402
from probe_readout_dial import DOC as DOC_FR         # noqa: E402
from probe_abstain_gen import DOC as DOC_RUCHER      # noqa: E402


def ghost(mem, ids, steps=8, min_count=None):
    """Follow the cold store's own successors. No model is involved."""
    mn = core.COLD_MIN_COUNT if min_count is None else min_count
    out = []
    hist = list(ids)
    for _ in range(steps):
        if len(hist) < core.NGRAM:
            break
        gram = np.array(hist[-core.NGRAM:], dtype=np.int32).tobytes()
        slot = mem.cold.get(gram)
        if slot is None or sum(slot[1].values()) < mn:
            break
        nxt = max(slot[1], key=slot[1].get)
        out.append(nxt)
        hist.append(nxt)
    return out


def main():
    tmp = tempfile.mkdtemp(prefix="ghost_")
    res = {}
    try:
        s = Sillage(model="qwen", state=tmp, quiet=True)
        text = Sillage.reflow(DOC_FR)
        held = Sillage.reflow(DOC_RUCHER)       # never read
        for _ in range(2):
            s.read_text(text)
        tok = s.load_tokenizer()
        mem = s.mem
        res["state"] = {"tokens": mem.tokens, "grams": len(mem.cold)}
        print(f"state: {res['state']}", flush=True)

        for label, body, key in (("read", text, "in_corpus"),
                                 ("never read", held, "off_corpus")):
            ids = tok.encode(body)
            offered = exact = 0
            lens, times = [], []
            # every position with enough history, stepping by 3 so the
            # sample is not dominated by one paragraph
            for i in range(core.NGRAM, len(ids) - 10, 3):
                t0 = time.perf_counter()
                g = ghost(mem, ids[:i])
                times.append((time.perf_counter() - t0) * 1000.0)
                if len(g) >= 2:
                    offered += 1
                    lens.append(len(g))
                    exact += g == list(ids[i:i + len(g)])
            n = len(times)
            row = {"positions": n,
                   "offered": offered,
                   "coverage": round(offered / max(1, n), 3),
                   "verbatim": exact,
                   "verbatim_share": round(exact / max(1, offered), 3),
                   "mean_len": round(float(np.mean(lens)), 1) if lens else 0,
                   "ms_median": round(float(np.median(times)), 4),
                   "ms_p99": round(float(np.percentile(times, 99)), 4)}
            res[key] = row
            print(f"  {label:<10} coverage {row['coverage']:.0%} "
                  f"({offered}/{n})  verbatim {row['verbatim_share']:.0%}  "
                  f"mean {row['mean_len']} tokens  "
                  f"{row['ms_median']:.3f} ms median / "
                  f"{row['ms_p99']:.3f} ms p99", flush=True)

        # what it actually looks like, on real prefixes
        ids = tok.encode(text)
        samples = []
        for i in (60, 120, 180, 240, 300):
            g = ghost(mem, ids[:i])
            if g:
                samples.append({"prefix": tok.decode(ids[max(0, i - 12):i]),
                                "ghost": tok.decode(g)})
        res["samples"] = samples
        print("", flush=True)
        for smp in samples:
            print(f"  ...{smp['prefix']!r} -> {smp['ghost']!r}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ic, oc = res["in_corpus"], res["off_corpus"]
    res["verdict"] = {
        "H1_latency_ms": ic["ms_p99"], "H1_holds": ic["ms_p99"] < 5.0,
        "H2_coverage": ic["coverage"], "H2_holds": ic["coverage"] >= 0.40,
        "H3_verbatim": ic["verbatim_share"], "H3_holds":
            ic["verbatim_share"] == 1.0,
        "H4_off_corpus": oc["coverage"], "H4_holds": oc["coverage"] < 0.25}
    print("\n" + json.dumps(res["verdict"], indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "ghost.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
