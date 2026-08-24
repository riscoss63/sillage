"""Streaming memories over a frozen LM's dumps: kNN-LM, byte-capped kNN-LM,
and the BHD Hebbian memory. Pure NumPy; every method consumes exactly the
same dumps (hidden states + base log-probs) produced by dump_base.py.

Protocol (per domain stream of N tokens, N-1 prediction positions):
  - online: at position j the memory may only contain entries written at
    positions i < j; after scoring x[j+1], the pair (h_j, x[j+1]) is written.
  - dev = first 20% of positions (hyperparameter selection),
    test = last 80% (reported).
  - all methods interpolate p = lam * p_mem + (1 - lam) * p_base and may
    abstain (use base only) when their retrieval-confidence signal is below
    a threshold tuned on dev.

Methods:
  knn      : faithful kNN-LM (Khandelwal et al. 2020) over the stream;
             exact keys, top-k softmax over -d/tau. Memory grows O(N).
  knn_cap  : same, but the store is capped at the SAME byte budget as BHD;
             entries kept by highest base-surprise (greedy eviction).
  bhd      : Hebbian outer-product memory M (D_k x D_v, fixed size).
             write: M += g * outer(q_j, V_tok[x_{j+1}]),
                    g = clip(base NLL, 0, CAP)   (surprise gating)
             read : p_mem = softmax(beta * V_tok @ (M.T @ q_j)).
             Keys q_j: random projection of the (causally centered,
             normalized) hidden state; dense-normalized or sparse-ternary
             (SDM-style top-w) variants.
  bhd_unif : ablation - same total write mass, no surprise gating.
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

import numpy as np

DATA, DUMPS, RESULTS = "data", "dumps", "results"
DEV_FRAC = 0.20
KNN_K = 32
D_K, D_V, W_SPARSE, CAP = 8192, 256, 192, 5.0
BETAS = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
LAMS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85]
THRESH_Q = [None, 0.25, 0.5, 0.75]    # abstain below this dev-quantile of conf
BHD_BYTES = D_K * D_V * 4


def load_domain(d):
    ids = np.load(os.path.join(DATA, f"{d}_ids.npy"))
    H = np.load(os.path.join(DUMPS, f"{d}_h.npy")).astype(np.float32)
    LP = np.load(os.path.join(DUMPS, f"{d}_lp.npy"))
    H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    vals = ids[1:].astype(np.int64)      # value of key j is x[j+1]
    return ids, H, LP, vals


def splits(n_pred):
    n_dev = int(DEV_FRAC * n_pred)
    dev = np.zeros(n_pred, dtype=bool)
    dev[:n_dev] = True
    return dev, ~dev


# ---------------------------------------------------------------- kNN-LM ----

def knn_neighbors(H, vals, block=1024, k=KNN_K):
    """Causal top-k neighbors for every position. Returns (dist, val) arrays
    of shape (n, k), padded with +inf / -1 where fewer neighbors exist."""
    n = len(H)
    nd = np.full((n, k), np.inf, dtype=np.float32)
    nv = np.full((n, k), -1, dtype=np.int64)
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        sims = H[b0:b1] @ H[:b1].T                     # (B, b1)
        for j in range(b0, b1):
            m = j - b0
            if j == 0:
                continue
            row = sims[m, :j]
            kk = min(k, j)
            idx = np.argpartition(-row, kk - 1)[:kk]
            d = np.sqrt(np.maximum(2.0 - 2.0 * row[idx], 0.0))
            nd[j, :kk] = d
            nv[j, :kk] = vals[idx]
    return nd, nv


def knn_capped_neighbors(H, vals, g, max_entries, k=KNN_K):
    """Online byte-capped store: keep highest-surprise entries."""
    n = len(H)
    nd = np.full((n, k), np.inf, dtype=np.float32)
    nv = np.full((n, k), -1, dtype=np.int64)
    keys = np.zeros((max_entries, H.shape[1]), dtype=np.float32)
    kv = np.zeros(max_entries, dtype=np.int64)
    kp = np.zeros(max_entries, dtype=np.float32)
    size = 0
    for j in range(n):
        if size > 0:
            row = keys[:size] @ H[j]
            kk = min(k, size)
            idx = np.argpartition(-row, kk - 1)[:kk] if size > kk else np.arange(size)
            d = np.sqrt(np.maximum(2.0 - 2.0 * row[idx], 0.0))
            order = np.argsort(d)[:kk]
            nd[j, :len(order)] = d[order]
            nv[j, :len(order)] = kv[idx][order]
        # write
        if size < max_entries:
            keys[size], kv[size], kp[size] = H[j], vals[j], g[j]
            size += 1
        else:
            w = int(np.argmin(kp[:size]))
            if g[j] > kp[w]:
                keys[w], kv[w], kp[w] = H[j], vals[j], g[j]
    return nd, nv


def knn_p_true(nd, nv, true_vals, k, tau):
    d = nd[:, :k]
    v = nv[:, :k]
    w = np.where(np.isfinite(d), np.exp(-d / tau), 0.0)
    den = w.sum(axis=1)
    num = (w * (v == true_vals[:, None])).sum(axis=1)
    p = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)
    conf = -np.where(np.isfinite(d[:, 0]), d[:, 0], np.inf)   # -min distance
    return p, conf


def unigram_cache_p_true(ids, vals, vocab=50257):
    """Null model: causal unigram cache of the stream. Any memory must beat
    this to claim CONTEXTUAL retrieval rather than trivial domain adaptation."""
    counts = np.zeros(vocab, dtype=np.float64)
    p = np.zeros(len(vals), dtype=np.float32)
    tot = 0
    for j in range(len(vals)):
        counts[ids[j]] += 1.0
        tot += 1
        p[j] = counts[vals[j]] / tot
    return p


# ------------------------------------------------------------ BHD memory ----

def bhd_pass(H, vals, g, seed=7, d_k=D_K, d_v=D_V, sparse_w=None,
             decay_per_token=1.0, vocab=50257):
    """One streaming pass. Returns per-position s_true, lse per beta, smax.

    Retrieval scores are normalized by ||u|| (u = M.T q) so they live in
    [-1, 1] regardless of how much has been written: beta stays calibrated
    over the whole stream, and smax is a scale-free confidence signal.
    """
    rng = np.random.default_rng(seed)
    R = (rng.integers(0, 2, size=(d_k, H.shape[1])) * 2.0 - 1.0).astype(np.float32)
    V = ((rng.integers(0, 2, size=(vocab, d_v)) * 2.0 - 1.0)
         / np.sqrt(d_v)).astype(np.float32)
    M = np.zeros((d_k, d_v), dtype=np.float32)
    n = len(H)
    s_true = np.zeros(n, dtype=np.float32)
    smax = np.zeros(n, dtype=np.float32)
    lse = np.zeros((n, len(BETAS)), dtype=np.float32)
    mu = np.zeros(H.shape[1], dtype=np.float32)
    betas = np.array(BETAS, dtype=np.float32)
    for j in range(n):
        c = H[j] - mu
        z = R @ c
        if sparse_w:
            idx = np.argpartition(-np.abs(z), sparse_w - 1)[:sparse_w]
            qs = (np.sign(z[idx]) / np.sqrt(sparse_w)).astype(np.float32)
            u = M[idx].T @ qs                 # (d_v,) sparse read
        else:
            q = z / (np.linalg.norm(z) + 1e-8)
            u = M.T @ q
        s = (V @ u) / (np.linalg.norm(u) + 1e-8)   # (vocab,) in [-1, 1]
        s_true[j] = s[vals[j]]
        smax[j] = s.max()
        m = betas[:, None] * s[None, :]            # (n_beta, vocab)
        mx = m.max(axis=1)
        lse[j] = mx + np.log(np.exp(m - mx[:, None]).sum(axis=1))
        if decay_per_token != 1.0:
            M *= decay_per_token
        gv = (g[j] * V[vals[j]]).astype(np.float32)
        if sparse_w:
            M[idx] += qs[:, None] * gv[None, :]    # sparse write
        else:
            M += q[:, None] * gv[None, :]
        mu += (H[j] - mu) / (j + 1)                # causal running mean
    return s_true, smax, lse


def bhd_p_true(s_true, lse, beta_idx, beta):
    return np.exp(beta * s_true - lse[:, beta_idx])


# ------------------------------------------------------------- evaluation ---

def tune_and_eval(p_mem_fn, confs, LP, dev, test, grid):
    """p_mem_fn(cfg) -> (p_mem_true, conf). Tune (cfg, lam, thresh) on dev
    NLL, return dev/test metrics for the winner."""
    p_base = np.exp(LP)
    best = None
    for cfg in grid:
        p_mem, conf = p_mem_fn(cfg)
        for tq in THRESH_Q:
            if tq is None:
                mask = np.ones(len(LP), dtype=bool)
            else:
                thr = np.quantile(conf[dev][np.isfinite(conf[dev])], tq) \
                    if np.isfinite(conf[dev]).any() else np.inf
                mask = conf >= thr
            for lam in LAMS:
                p = np.where(mask, lam * p_mem + (1 - lam) * p_base, p_base)
                nll_dev = float(-np.log(np.maximum(p[dev], 1e-30)).mean())
                if best is None or nll_dev < best["nll_dev"]:
                    nll_test = float(-np.log(np.maximum(p[test], 1e-30)).mean())
                    best = {"cfg": str(cfg), "lam": lam, "thresh_q": tq,
                            "nll_dev": nll_dev, "nll_test": nll_test,
                            "p_true_test": p[test].copy()}
    return best


def block_bootstrap_dnll(p_test, lp_base_test, block=512, reps=1000, seed=3):
    """95% CI of (base NLL - method NLL) on test, block-resampled."""
    d = (-lp_base_test) - (-np.log(np.maximum(p_test, 1e-30)))
    nb = max(1, len(d) // block)
    blocks = [d[i * block:(i + 1) * block] for i in range(nb)]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([blocks[i] for i in
                             rng.integers(0, nb, size=nb)]).mean()
             for _ in range(reps)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def eval_domain(d, verbose=True):
    ids, H, LP, vals = load_domain(d)
    n = len(H)
    dev, test = splits(n)
    g = np.clip(-LP, 0.0, CAP)
    out = {"domain": d, "n_pred": int(n),
           "base_nll_test": float(-LP[test].mean()),
           "base_ppl_test": float(np.exp(-LP[test].mean()))}

    def report(name, best, bytes_):
        gain = out["base_nll_test"] - best["nll_test"]
        ci = block_bootstrap_dnll(best.pop("p_true_test"), LP[test])
        out[name] = {**best, "ppl_test": float(np.exp(best["nll_test"])),
                     "dnll_test": float(gain), "dnll_ci95": ci,
                     "memory_bytes": int(bytes_)}
        if verbose:
            print(f"  {name}: test NLL {best['nll_test']:.4f} "
                  f"(base {out['base_nll_test']:.4f}, dNLL {gain:+.4f} "
                  f"CI {ci}) cfg={best['cfg']} lam={best['lam']} "
                  f"thr={best['thresh_q']} bytes={bytes_:,}", flush=True)

    # kNN-LM (unbounded)
    nd, nv = knn_neighbors(H, vals)
    grid = [(k, tau) for k in (8, 16, 32) for tau in (0.5, 1.0, 3.0)]
    best = tune_and_eval(lambda c: knn_p_true(nd, nv, vals, c[0], c[1]),
                         None, LP, dev, test, grid)
    report("knn", best, n * (768 * 2 + 4))

    # byte-capped kNN at the BHD budget
    cap_entries = BHD_BYTES // (768 * 2 + 4 + 4)
    nd2, nv2 = knn_capped_neighbors(H, vals, g, cap_entries)
    best = tune_and_eval(lambda c: knn_p_true(nd2, nv2, vals, c[0], c[1]),
                         None, LP, dev, test, grid)
    report("knn_cap", best, BHD_BYTES)

    # unigram cache: the null model every memory must beat
    p_uni = unigram_cache_p_true(ids, vals)
    best = tune_and_eval(lambda c: (p_uni, p_uni), None, LP, dev, test,
                         [("uni",)])
    report("cache_unigram", best, 50257 * 4)

    # BHD: dense and sparse key variants (surprise-gated)
    for name, sw in [("bhd_dense", None), ("bhd_sparse", W_SPARSE)]:
        s_true, smax, lse = bhd_pass(H, vals, g, sparse_w=sw)
        best = tune_and_eval(
            lambda c: (bhd_p_true(s_true, lse, c[0], BETAS[c[0]]), smax),
            None, LP, dev, test, [(i,) for i in range(len(BETAS))])
        report(name, best, BHD_BYTES)

    # ablation: uniform writes, same total mass (best key variant = sparse)
    g_unif = np.full_like(g, float(g.mean()))
    s_true, smax, lse = bhd_pass(H, vals, g_unif, sparse_w=W_SPARSE)
    best = tune_and_eval(
        lambda c: (bhd_p_true(s_true, lse, c[0], BETAS[c[0]]), smax),
        None, LP, dev, test, [(i,) for i in range(len(BETAS))])
    report("bhd_unif", best, BHD_BYTES)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"memories_{d}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import sys
    domains = sys.argv[1:] or ["relativity", "alice", "bhd"]
    for d in domains:
        print(f"=== {d} ===", flush=True)
        eval_domain(d)
