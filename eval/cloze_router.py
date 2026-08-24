"""Downstream content-token recall with the ROUTER memory (G + S blocks).

Same item selection and protocol as cloze_eval.py (identical items, so the
numbers are directly comparable to the paper-1 table); predictions compare
frozen LM | + G-only (paper-1 Sillage) | + router. Interpolation parameters are
read from the domain's router JSON (perplexity-tuned; no task tuning);
per-block abstention thresholds are recomputed as calib-quantile values.

Usage: python cloze_router.py <gpt2|qwen> <domain> [cap]
Output: results/cloze_router_<prefix><domain>.json
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
import re
import sys

import numpy as np

from memories import BETAS, CAP
from sillage_router import B_LIST_MULTI
from sillage_semantic import (B_BITS, D_BAND, D_G, D_V, EPS_EIG, L_BANDS,
                           NGRAM, WHITEN_FIT, band_vector_cache)

TOPK = 128
MAX_ITEMS = 2000
CALIB = 2000
WORD = re.compile(r"^ [^\W\d_]{5,}$", re.UNICODE)


def parse_cfg(s):
    m = re.match(r"\('sm', (\d+)\)", s)
    return ("sm", int(m.group(1))) if m else ("quad", -1)


def read_params(prefix, domain):
    j = json.load(open(f"results/sillage_router_{prefix}{domain}_multi.json"))
    g = j["g_only"]
    r = j["router"]
    return {"G": {"cfg": parse_cfg(g["cfg"]), "lam": g["lam"],
                  "thr": g["thr"]},
            "S": {"cfg": parse_cfg(r.get("cfg", r.get("cfg_S"))),
                  "lam": r.get("lam", r.get("lam_S")),
                  "thr": r.get("thr", r.get("thr_S"))}}


def main():
    which, domain = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    prefix = "" if which == "gpt2" else "q_"
    vocab = 50257 if which == "gpt2" else 151_936
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "openai-community/gpt2" if which == "gpt2" else "Qwen/Qwen3-0.6B")
    params = read_params(prefix, domain)
    ids_all = np.load(f"data/{prefix}{domain}_ids.npy")
    LP = np.load(f"dumps/{prefix}{domain}_lp.npy")
    TI = np.load(f"dumps/{prefix}{domain}_topi.npy")
    TL = np.load(f"dumps/{prefix}{domain}_topl.npy").astype(np.float32)
    H = np.load(f"dumps/{prefix}{domain}_h.npy", mmap_mode="r")
    n = min(len(LP), cap)
    ids = ids_all[: n + 1]
    vals = ids[1:].astype(np.int64)
    n_dev = int(0.2 * n)

    # ---- items (identical rule to cloze_eval) -------------------------------
    word_ok, counts = {}, np.zeros(vocab, dtype=np.int32)
    cand, prior = [], []
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
    need = np.zeros(n, dtype=np.int8)
    need[calib] = 1
    need[cand] = 2
    item_of = {int(j): i for i, j in enumerate(cand)}
    print(f"{prefix}{domain}: {len(cand)} items", flush=True)

    # ---- router pass with per-item stores -----------------------------------
    rngV = np.random.default_rng(7001)
    rngT = np.random.default_rng(7002)
    rngW = np.random.default_rng(7003)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)
    Wh = rngW.normal(size=(H.shape[1], L_BANDS * B_BITS)).astype(np.float32)
    getv = band_vector_cache()
    D_S = len(B_LIST_MULTI) * L_BANDS * D_BAND
    M_G = np.zeros((D_G, D_V), dtype=np.float32)
    M_S = np.zeros((D_S, D_V), dtype=np.float32)
    ni = len(cand)
    store = {b: {"ids": np.zeros((ni, TOPK), np.int64),
                 "s_top": np.zeros((ni, TOPK), np.float32),
                 "s_base": np.zeros((ni, TOPK), np.float32),
                 "lse": np.zeros(ni, np.float32),
                 "smax": np.zeros(ni, np.float32)} for b in ("G", "S")}
    beta = {b: BETAS[params[b]["cfg"][1]] for b in ("G", "S")}
    smax_cal = {"G": [], "S": []}
    g_raw = np.ones(D_G, dtype=np.float32)
    invG = 1.0 / np.sqrt(D_G)
    mu, W_zca = None, None
    pw2 = 2 ** np.arange(B_BITS)

    def full_score(u, block, j):
        un = float(np.linalg.norm(u)) + 1e-8
        s = (V @ u) / un
        if need[j] == 1:
            smax_cal[block].append(float(s.max()))
        else:
            i = item_of[j]
            d = store[block]
            top = np.argpartition(-s, TOPK - 1)[:TOPK]
            d["ids"][i] = top
            d["s_top"][i] = s[top]
            d["s_base"][i] = s[TI[j]]
            m = beta[block] * s
            mx = m.max()
            d["lse"][i] = mx + np.log(np.exp(m - mx).sum())
            d["smax"][i] = s.max()

    for j in range(n):
        g_raw = np.roll(g_raw, 1)
        g_raw *= T[ids[j]]
        if j >= NGRAM:
            g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
        qG = g_raw * invG
        uG = M_G.T @ qG
        if j == WHITEN_FIT:
            X = np.array(H[:j], dtype=np.float32)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            mu = X.mean(0)
            C = X - mu
            cov = (C.T @ C) / len(C)
            w, U = np.linalg.eigh(cov)
            W_zca = (U @ np.diag(1.0 / np.sqrt(w + EPS_EIG)) @ U.T
                     ).astype(np.float32)
        qS, uS = None, None
        if W_zca is not None:
            h = np.array(H[j], dtype=np.float32)
            h /= np.linalg.norm(h) + 1e-8
            z = (h - mu) @ W_zca
            bits = ((z @ Wh) > 0).reshape(L_BANDS, B_BITS)
            qS = np.empty(D_S, dtype=np.float32)
            scale = 1.0 / np.sqrt(len(B_LIST_MULTI) * L_BANDS * D_BAND)
            slot = 0
            for gi, b in enumerate(B_LIST_MULTI):
                for k in range(L_BANDS):
                    pat = int(bits[k, :b] @ pw2[:b])
                    qS[slot * D_BAND:(slot + 1) * D_BAND] = \
                        scale * getv(gi * L_BANDS + k, pat)
                    slot += 1
            uS = M_S.T @ qS
        if need[j]:
            full_score(uG, "G", j)
            if uS is not None:
                full_score(uS, "S", j)
        g = min(CAP, max(0.0, -float(LP[j])))
        a = max(0.0, float(uG @ V[vals[j]]))
        M_G += (np.sqrt(a * a + g) - a) * qG[:, None] * V[vals[j]][None, :]
        if uS is not None:
            a = max(0.0, float(uS @ V[vals[j]]))
            M_S += (np.sqrt(a * a + g) - a) * qS[:, None] * V[vals[j]][None, :]

    thr = {}
    for b in ("G", "S"):
        tq = params[b]["thr"]
        arr = np.array(smax_cal[b])
        arr = arr[arr > 0] if b == "S" else arr
        thr[b] = -np.inf if tq is None else float(np.quantile(arr, tq))

    # ---- predictions --------------------------------------------------------
    correct = {"base": [], "gonly": [], "router": []}
    lamG, lamS = params["G"]["lam"], params["S"]["lam"]
    for i, j in enumerate(cand):
        truth = int(vals[j])
        base_pred = int(TI[j, 0])
        correct["base"].append(base_pred == truth)
        p_base = dict(zip(TI[j].tolist(), np.exp(TL[j]).tolist()))
        pmap = {}
        for b in ("G", "S"):
            d = store[b]
            m = {}
            if d["smax"][i] >= thr[b]:
                for t, sv in zip(d["ids"][i].tolist(),
                                 d["s_top"][i].tolist()):
                    m[int(t)] = float(np.exp(beta[b] * sv - d["lse"][i]))
                for t, sv in zip(TI[j].tolist(), d["s_base"][i].tolist()):
                    m.setdefault(int(t),
                                 float(np.exp(beta[b] * sv - d["lse"][i])))
            pmap[b] = m
        # G-only prediction
        if pmap["G"]:
            cands = set(p_base) | set(pmap["G"])
            pred = max(cands, key=lambda t: lamG * pmap["G"].get(t, 0.0)
                       + (1 - lamG) * p_base.get(t, 0.0))
        else:
            pred = base_pred
        correct["gonly"].append(pred == truth)
        # router prediction
        aG, aS = bool(pmap["G"]), bool(pmap["S"])
        if aG or aS:
            lg = lamG if aG else 0.0
            ls = lamS if aS else 0.0
            rest = max(1 - lg - ls, 0.0)
            cands = set(p_base) | set(pmap["G"]) | set(pmap["S"])
            pred = max(cands, key=lambda t: lg * pmap["G"].get(t, 0.0)
                       + ls * pmap["S"].get(t, 0.0)
                       + rest * p_base.get(t, 0.0))
        else:
            pred = base_pred
        correct["router"].append(pred == truth)

    def acc(key, mask=None):
        c = np.array(correct[key])
        if mask is not None:
            c = c[mask]
        k, N = int(c.sum()), len(c)
        p = k / N
        z = 1.96
        den = 1 + z * z / N
        ctr = (p + z * z / (2 * N)) / den
        hw = z * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / den
        return {"acc": round(p, 4), "n": N,
                "wilson95": [round(ctr - hw, 4), round(ctr + hw, 4)]}

    out = {"model": which, "domain": domain, "items": int(len(cand)),
           "params": {b: {"cfg": str(params[b]["cfg"]),
                          "lam": params[b]["lam"],
                          "thr": params[b]["thr"]} for b in ("G", "S")},
           "all": {k: acc(k) for k in correct},
           "recurring_ge5": {k: acc(k, prior >= 5) for k in correct}}
    b = np.array(correct["base"])
    r = np.array(correct["router"])
    out["mcnemar_router_vs_base"] = {"router_only": int((r & ~b).sum()),
                                     "base_only": int((b & ~r).sum())}
    with open(f"results/cloze_router_{prefix}{domain}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ["all", "recurring_ge5",
                                          "mcnemar_router_vs_base"]},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
