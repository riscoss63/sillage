"""Downstream evaluation: content-token recall after one reading pass.

Task: at automatically selected test positions whose target is a content
word (leading-space alphabetic piece, >= 5 characters, token type already
seen >= 2 times earlier in the stream), predict the next token by greedy
argmax. Compared systems: frozen LM | LM + kNN-LM | LM + Sillage, each using
the interpolation weights tuned on that stream's dev split for perplexity
(no cloze-specific tuning). Candidates for the mixed argmax are the union
of the base model's top-128 and the memory's top-128 (base probability of
out-of-top-128 candidates approximated as 0; stated in the paper).

Also reports the "recurring" subset (target type seen >= 5 times) and
example predictions for the appendix.

Usage: python cloze_eval.py <gpt2|qwen> <domain>
Output: results/cloze_<prefix><domain>.json
"""


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
import re
import sys

import numpy as np

from memories import CAP

TOPK = 128
D_G, D_V = 4096, 256
BETAS = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
MAX_ITEMS = 2000
CALIB = 2000
WORD = re.compile(r"^ [^\W\d_]{5,}$", re.UNICODE)

CONFIGS = {   # (beta_idx, lam, thr_q) for Sillage ; (k, tau, lam, thr_q) for kNN
    ("", "bhd"): {"sillage": (4, 0.3, 0.75), "knn": (32, 0.5, 0.1, 0.25)},
    ("", "tolstoy"): {"sillage": (5, 0.02, 0.75), "knn": (32, 0.5, 0.05, None)},
}


def qwen_config():
    j = json.load(open("results/model2_qwen_bhd.json"))
    cfg = j["sillage_amp_system"]
    m = re.match(r"\('sm', (\d+)\)", cfg["cfg"])
    beta_idx = int(m.group(1)) if m else 4
    kc = j["knn"]
    km = re.match(r"\((\d+), ([\d.]+)\)", kc["cfg"])
    return {"sillage": (beta_idx, cfg["lam"], cfg["thresh_q"]),
            "knn": (int(km.group(1)), float(km.group(2)), kc["lam"],
                    kc["thresh_q"])}


def main(which, domain):
    prefix = "" if which == "gpt2" else "q_"
    vocab = 50257 if which == "gpt2" else 151_936
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "openai-community/gpt2" if which == "gpt2" else "Qwen/Qwen3-0.6B")

    ids = np.load(f"data/{prefix}{domain}_ids.npy")
    LP = np.load(f"dumps/{prefix}{domain}_lp.npy")
    TI = np.load(f"dumps/{prefix}{domain}_topi.npy")
    vals = ids[1:].astype(np.int64)
    n = len(vals)
    cfg = CONFIGS.get((prefix, domain)) or qwen_config()
    b_idx, lam_s, thr_s = cfg["sillage"]
    k_knn, tau_knn, lam_k, thr_k = cfg["knn"]

    # ---- item selection (causal type counts) --------------------------------
    word_ok = {}
    counts = np.zeros(vocab, dtype=np.int32)
    cand, prior = [], []
    n_dev = int(0.2 * n)
    for j in range(n):
        t = int(vals[j])
        if j >= max(n_dev, 256):
            ok = word_ok.get(t)
            if ok is None:
                ok = bool(WORD.match(tok.decode([t])))
                word_ok[t] = ok
            if ok and counts[t] >= 2:
                cand.append(j)
                prior.append(int(counts[t]))
        counts[int(ids[j])] += 1
    cand = np.array(cand)
    prior = np.array(prior)
    if len(cand) > MAX_ITEMS:
        sel = np.linspace(0, len(cand) - 1, MAX_ITEMS).astype(int)
        cand, prior = cand[sel], prior[sel]
    calib = np.linspace(256, n_dev - 1, CALIB).astype(int)
    print(f"{prefix}{domain}: {len(cand)} items "
          f"(recurring>=5: {(prior >= 5).sum()})", flush=True)

    # ---- Sillage pass (amp values, model gate, n=4), scoring at items+calib ----
    rngV = np.random.default_rng(7001)
    rngT = np.random.default_rng(7002)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    M = np.zeros((D_G, D_V), dtype=np.float32)
    beta = BETAS[b_idx]
    need = np.zeros(n, dtype=np.int8)
    need[calib] = 1
    need[cand] = 2
    item_of = {int(j): i for i, j in enumerate(cand)}
    mem_ids = np.zeros((len(cand), TOPK), dtype=np.int64)
    s_mem = np.zeros((len(cand), TOPK), dtype=np.float32)
    s_base = np.zeros((len(cand), TOPK), dtype=np.float32)
    lse_it = np.zeros(len(cand), dtype=np.float32)
    smax_it = np.zeros(len(cand), dtype=np.float32)
    smax_cal = []
    g_raw = np.ones(D_G, dtype=np.float32)
    inv = 1.0 / np.sqrt(D_G)
    for j in range(n):
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= 4:
            g_raw *= np.roll(T[ids[j - 4]], 4)
        q = g_raw * inv
        u = M.T @ q
        if need[j]:
            un = float(np.linalg.norm(u)) + 1e-8
            s = (V @ u) / un
            if need[j] == 1:
                smax_cal.append(float(s.max()))
            else:
                i = item_of[j]
                top = np.argpartition(-s, TOPK - 1)[:TOPK]
                mem_ids[i] = top
                s_mem[i] = s[top]
                s_base[i] = s[TI[j]]
                m = beta * s
                mx = m.max()
                lse_it[i] = mx + np.log(np.exp(m - mx).sum())
                smax_it[i] = s.max()
        a = max(0.0, float(u @ V[vals[j]]))
        gj = min(CAP, max(0.0, -float(LP[j])))
        M += (np.sqrt(a * a + gj) - a) * q[:, None] * V[vals[j]][None, :]
    thr_val = (np.quantile(smax_cal, thr_s) if thr_s is not None else -np.inf)

    # ---- kNN at items (+calib for threshold) --------------------------------
    H = np.load(f"dumps/{prefix}{domain}_h.npy").astype(np.float32)
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    def knn_at(positions):
        nd = np.full((len(positions), k_knn), np.inf, dtype=np.float32)
        nv = np.full((len(positions), k_knn), -1, dtype=np.int64)
        for m0 in range(0, len(positions), 256):
            m1 = min(m0 + 256, len(positions))
            hi = int(positions[m1 - 1])
            sims = H[positions[m0:m1]] @ H[:hi].T
            for m in range(m0, m1):
                j = int(positions[m])
                row = sims[m - m0, :j]
                kk = min(k_knn, j)
                idx = np.argpartition(-row, kk - 1)[:kk]
                nd[m, :kk] = np.sqrt(np.maximum(2 - 2 * row[idx], 0))
                nv[m, :kk] = vals[idx]
        return nd, nv
    nd_i, nv_i = knn_at(cand)
    nd_c, _ = knn_at(calib)
    conf_cal = -nd_c[:, 0]
    thr_knn = (np.quantile(conf_cal[np.isfinite(conf_cal)], thr_k)
               if thr_k is not None else -np.inf)
    del H

    # ---- predictions --------------------------------------------------------
    TL = np.load(f"dumps/{prefix}{domain}_topl.npy").astype(np.float32)
    correct = {"base": [], "knn": [], "sillage": []}
    examples = []
    for i, j in enumerate(cand):
        truth = int(vals[j])
        base_pred = int(TI[j, 0])
        correct["base"].append(base_pred == truth)
        p_base = dict(zip(TI[j].tolist(), np.exp(TL[j]).tolist()))
        # kNN
        w = np.where(np.isfinite(nd_i[i]), np.exp(-nd_i[i] / tau_knn), 0.0)
        den = w.sum()
        if -nd_i[i, 0] >= thr_knn and den > 0:
            p_knn = {}
            for t, ww in zip(nv_i[i].tolist(), w.tolist()):
                if t >= 0:
                    p_knn[t] = p_knn.get(t, 0.0) + ww / den
            cands = set(p_base) | set(p_knn)
            pred_k = max(cands, key=lambda t: lam_k * p_knn.get(t, 0.0)
                         + (1 - lam_k) * p_base.get(t, 0.0))
        else:
            pred_k = base_pred
        correct["knn"].append(pred_k == truth)
        # Sillage
        if smax_it[i] >= thr_val:
            pm = {int(t): float(np.exp(beta * sv - lse_it[i]))
                  for t, sv in zip(mem_ids[i], s_mem[i])}
            for t, sv in zip(TI[j].tolist(), s_base[i].tolist()):
                pm.setdefault(int(t), float(np.exp(beta * sv - lse_it[i])))
            cands = set(p_base) | set(pm)
            pred_s = max(cands, key=lambda t: lam_s * pm.get(t, 0.0)
                         + (1 - lam_s) * p_base.get(t, 0.0))
        else:
            pred_s = base_pred
        correct["sillage"].append(pred_s == truth)
        if pred_s == truth and base_pred != truth and len(examples) < 12:
            ctx = tok.decode(ids[max(0, j - 24): j + 1].tolist())
            examples.append({"context_tail": ctx[-120:],
                             "truth": tok.decode([truth]),
                             "base": tok.decode([base_pred]),
                             "sillage": tok.decode([pred_s])})

    def acc(mask=None):
        res = {}
        for m_name, c in correct.items():
            c = np.array(c)
            if mask is not None:
                c = c[mask]
            k, N = int(c.sum()), len(c)
            p = k / N
            z = 1.96
            den = 1 + z * z / N
            ctr = (p + z * z / (2 * N)) / den
            hw = z * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / den
            res[m_name] = {"acc": round(p, 4), "n": N,
                           "wilson95": [round(ctr - hw, 4),
                                        round(ctr + hw, 4)]}
        return res

    b = np.array(correct["base"]); s = np.array(correct["sillage"])
    out = {"model": which, "domain": domain, "items": int(len(cand)),
           "all": acc(), "recurring_ge5": acc(prior >= 5),
           "mcnemar_sillage_vs_base": {"sillage_only": int((s & ~b).sum()),
                                    "base_only": int((b & ~s).sum())},
           "config": cfg, "examples": examples}
    os.makedirs("results", exist_ok=True)
    with open(f"results/cloze_{prefix}{domain}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ["all", "recurring_ge5",
                                          "mcnemar_sillage_vs_base"]}, indent=1),
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
