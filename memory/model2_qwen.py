"""Second-model replication: Qwen3-0.6B (vocab 151,936, hidden 1024).

Same protocol as the GPT-2 experiments on the three short domains:
base | unigram cache | exact 4-gram dict | kNN-LM unbounded | kNN-LM
byte-matched to Sillage | Sillage count control | Sillage amp+system n=4
(+ amp+system n=2+4+8 on the two classics). Paired test Sillage vs unbounded
kNN on the manuscripts domain.

Usage: python model2_qwen.py <domain>       (bhd | relativity | alice)
Output: results/model2_qwen_<domain>.json
"""


# --- repo bootstrap (added by reorganize.py) ---
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
import sys

import numpy as np

from ngram_memory import ngram_dict_p_true
from sillage_factorial import p_true_of, v3_pass
from memories import (BETAS, CAP, block_bootstrap_dnll, knn_capped_neighbors,
                      knn_neighbors, knn_p_true, splits, tune_and_eval,
                      unigram_cache_p_true)

VOCAB = 151_936
SILLAGE_BYTES = 4096 * 256 * 4


def load_q(d):
    ids = np.load(f"data/q_{d}_ids.npy")
    H = np.load(f"dumps/q_{d}_h.npy").astype(np.float32)
    LP = np.load(f"dumps/q_{d}_lp.npy")
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    return ids, H, LP, ids[1:].astype(np.int64)


def run(d):
    ids, H, LP, vals = load_q(d)
    n = len(H)
    dev, test = splits(n)
    g = np.clip(-LP, 0.0, CAP)
    hd = H.shape[1]
    out = {"domain": d, "model": "Qwen3-0.6B", "n_pred": int(n),
           "base_nll_test": float(-LP[test].mean()),
           "base_ppl_test": float(np.exp(-LP[test].mean()))}
    print(f"q_{d}: base NLL {out['base_nll_test']:.4f} "
          f"(PPL {out['base_ppl_test']:.1f})", flush=True)
    p_saved = {}

    def report(name, best, bytes_):
        p_test = best.pop("p_true_test")
        p_saved[name] = p_test
        gain = out["base_nll_test"] - best["nll_test"]
        ci = block_bootstrap_dnll(p_test, LP[test])
        out[name] = {**best, "dnll_test": float(gain), "dnll_ci95": ci,
                     "memory_bytes": int(bytes_)}
        print(f"  {name}: dNLL {gain:+.4f} CI {ci} cfg={best['cfg']} "
              f"lam={best['lam']} thr={best['thresh_q']}", flush=True)

    # cache + dict
    p_uni = unigram_cache_p_true(ids, vals, vocab=VOCAB)
    report("cache_unigram", tune_and_eval(lambda c: (p_uni, p_uni), None, LP,
                                          dev, test, [("uni",)]), VOCAB * 4)
    p_ng, conf_ng, n_pairs = ngram_dict_p_true(ids, vals)
    best = tune_and_eval(lambda c: (p_ng, conf_ng), None, LP, dev, test,
                         [("dict4",)])
    out["ngram_dict_pairs"] = int(n_pairs)
    report("ngram_dict", best, n_pairs * 16)

    # kNN unbounded + byte-matched
    nd, nv = knn_neighbors(H, vals)
    grid_k = [(k, tau) for k in (8, 16, 32) for tau in (0.5, 1.0, 3.0)]
    report("knn", tune_and_eval(lambda c: knn_p_true(nd, nv, vals, c[0], c[1]),
                                None, LP, dev, test, grid_k),
           n * (hd * 2 + 4))
    cap_entries = SILLAGE_BYTES // (hd * 2 + 4 + 4)
    nd2, nv2 = knn_capped_neighbors(H, vals, g, cap_entries)
    report("knn_cap_matched",
           tune_and_eval(lambda c: knn_p_true(nd2, nv2, vals, c[0], c[1]),
                         None, LP, dev, test, grid_k), SILLAGE_BYTES)

    # Sillage
    grid_b = [("sm", i) for i in range(len(BETAS))] + [("quad", -1)]
    variants = [("sillage_count", dict(value="count", gate="model",
                                    scales=(4,)))]
    variants.append(("sillage_amp_system", dict(value="amp", gate="system",
                                             scales=(4,))))
    if d in ("relativity", "alice"):
        variants.append(("sillage_amp_system_n248",
                         dict(value="amp", gate="system", scales=(2, 4, 8))))
    for name, kw in variants:
        s_true, smax, lse, sumsq = v3_pass(ids, vals, LP, vocab=VOCAB, **kw)
        report(name, tune_and_eval(
            lambda c: (p_true_of(c, s_true, lse, sumsq), smax),
            None, LP, dev, test, grid_b),
            len(kw["scales"]) * 4096 * 256 * 4)

    # paired headline on the manuscripts domain
    if d == "bhd" and "sillage_amp_system" in p_saved and "knn" in p_saved:
        dd = (-np.log(np.maximum(p_saved["knn"][:], 1e-30))
              + np.log(np.maximum(p_saved["sillage_amp_system"][:], 1e-30)))
        nb = len(dd) // 512
        blocks = [dd[i * 512:(i + 1) * 512] for i in range(nb)]
        rng = np.random.default_rng(3)
        means = [np.concatenate([blocks[i] for i in
                                 rng.integers(0, nb, nb)]).mean()
                 for _ in range(2000)]
        out["paired_sillage_vs_knn"] = {
            "mean": float(dd.mean()),
            "ci95": [float(np.quantile(means, 0.025)),
                     float(np.quantile(means, 0.975))],
            "p_better": float((np.array(means) > 0).mean())}
        print(f"  paired Sillage vs kNN: {dd.mean():+.4f} "
              f"CI {out['paired_sillage_vs_knn']['ci95']} "
              f"P={out['paired_sillage_vs_knn']['p_better']:.3f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(f"results/model2_qwen_{d}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    run(sys.argv[1])
