"""Step-4 scaling & replication: long-horizon stability and a second model.

Same prequential protocol as fastweights_combo.py (score before update), with
  * selectable base model (gpt2 | qwen)
  * selectable adapter ranks (default: the winning r=16 uniform rule)
  * per-50k-token segment curves for FW, memory and their combination, so
    long-horizon STABILITY is visible, not just the aggregate

Usage: python fastweights_scale.py <gpt2|qwen> <domain> [r1,r2,...]
Output: results/fwscale_<prefix><domain>.json
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
import sys
import time

import numpy as np
import torch

from memories import BETAS, CAP, LAMS, THRESH_Q

D_G, D_V, NGRAM = 4096, 256, 4
WINDOW, STRIDE = 1024, 512
ETA = 0.1
SEG = 50_000
MODELS = {"gpt2": ("openai-community/gpt2", 50257, ""),
          "qwen": ("Qwen/Qwen3-0.6B", 151_936, "q_")}


@torch.no_grad()
def run_pass(which, domain, ranks):
    name, vocab, prefix = MODELS[which]
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    hidden = model.config.hidden_size
    ids = np.load(f"data/{prefix}{domain}_ids.npy").astype(np.int64)
    n = len(ids) - 1
    rng = np.random.default_rng(7010)
    Rf = {r: rng.normal(size=(hidden, r)).astype(np.float32) / np.sqrt(hidden)
          for r in ranks}
    rngV = np.random.default_rng(7001)
    rngT = np.random.default_rng(7002)
    V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
         / np.sqrt(D_V)).astype(np.float32)
    T = (rngT.integers(0, 2, size=(vocab, D_G), dtype=np.int8) * 2 - 1)

    A = {r: np.zeros((vocab, r), dtype=np.float32) for r in ranks}
    M_G = np.zeros((D_G, D_V), dtype=np.float32)
    lp_base = np.zeros(n, dtype=np.float32)
    lp_fw = {r: np.zeros(n, dtype=np.float32) for r in ranks}
    s_true = np.zeros(n, dtype=np.float32)
    smax = np.zeros(n, dtype=np.float32)
    lse = np.zeros((n, len(BETAS)), dtype=np.float32)
    betas = np.array(BETAS, dtype=np.float32)
    g_raw = np.ones(D_G, dtype=np.float32)
    invG = 1.0 / np.sqrt(D_G)
    g_sum, g_cnt = 0.0, 0
    x = torch.tensor(ids)
    a, done = 0, 0
    t0 = time.time()
    while a < n:
        w = min(WINDOW, len(ids) - a)
        out = model(x[a:a + w].unsqueeze(0), output_hidden_states=True)
        logits = out.logits[0].float().numpy()
        hs = out.hidden_states[-1][0].float().numpy()
        lo = 0 if a == 0 else WINDOW - STRIDE
        for i in range(lo, w):
            j = a + i
            if j >= n:
                break
            truth = int(ids[j + 1])
            lb = logits[i]
            mx = lb.max()
            lpb = lb - (mx + np.log(np.exp(lb - mx).sum()))
            lp_base[j] = lpb[truth]
            g = min(CAP, max(0.0, float(-lpb[truth])))
            g_sum += g
            g_cnt += 1
            g_mean = g_sum / g_cnt          # uniform step (gating loses)
            for r in ranks:
                phi = hs[i] @ Rf[r]
                phi /= np.linalg.norm(phi) + 1e-8
                la = lb + A[r] @ phi
                m = la.max()
                p = np.exp(la - m)
                p /= p.sum()
                lp_fw[r][j] = np.log(max(p[truth], 1e-30))
                step = ETA * g_mean
                A[r] -= step * np.outer(p, phi)
                A[r][truth] += step * phi
            g_raw = np.roll(g_raw, 1)
            g_raw *= T[ids[j]]
            if j >= NGRAM:
                g_raw *= np.roll(T[ids[j - NGRAM]], NGRAM)
            q = g_raw * invG
            u = M_G.T @ q
            un = float(np.linalg.norm(u)) + 1e-8
            s = (V @ u) / un
            s_true[j] = s[truth]
            smax[j] = s.max()
            mm = betas[:, None] * s[None, :]
            mmx = mm.max(axis=1)
            lse[j] = mmx + np.log(np.exp(mm - mmx[:, None]).sum(axis=1))
            amp = max(0.0, float(u @ V[truth]))
            M_G += (np.sqrt(amp * amp + g) - amp) * q[:, None] * V[truth][None, :]
            done += 1
        if done and done % 25000 < STRIDE:
            print(f"  ... {done}/{n} ({(time.time()-t0)/60:.0f} min)",
                  flush=True)
        if a + w >= len(ids):
            break
        a += STRIDE
    return lp_base, lp_fw, s_true, smax, lse


def tune_memory(p_basis, s_true, smax, lse, dev):
    best = None
    for bi in range(len(BETAS)):
        pm = np.exp(BETAS[bi] * s_true - lse[:, bi])
        for tq in THRESH_Q:
            mask = (np.ones(len(p_basis), bool) if tq is None
                    else smax >= np.quantile(smax[dev], tq))
            for lam in LAMS:
                p = np.where(mask, lam * pm + (1 - lam) * p_basis, p_basis)
                nd = float(-np.log(np.maximum(p[dev], 1e-30)).mean())
                if best is None or nd < best["nll_dev"]:
                    best = {"nll_dev": nd, "p": p,
                            "params": {"beta_idx": bi, "lam": lam,
                                       "thr_q": tq}}
    return best


def paired(pa, pb, test):
    d = (-np.log(np.maximum(pa[test], 1e-30))
         + np.log(np.maximum(pb[test], 1e-30)))
    nb = len(d) // 512
    blocks = [d[k * 512:(k + 1) * 512] for k in range(nb)]
    rb = np.random.default_rng(3)
    means = [np.concatenate([blocks[k] for k in rb.integers(0, nb, nb)]).mean()
             for _ in range(1000)]
    return {"mean": float(d.mean()),
            "ci95": [float(np.quantile(means, 0.025)),
                     float(np.quantile(means, 0.975))],
            "p_better": float((np.array(means) > 0).mean())}


def segments(lp_base, p, n):
    d = (-lp_base) - (-np.log(np.maximum(p, 1e-30)))
    return [float(d[s:s + SEG].mean()) for s in range(0, n, SEG)]


def main():
    which, domain = sys.argv[1], sys.argv[2]
    ranks = ([int(v) for v in sys.argv[3].split(",")]
             if len(sys.argv) > 3 else [16])
    prefix = MODELS[which][2]
    lp_base, lp_fw, s_true, smax, lse = run_pass(which, domain, ranks)
    n = len(lp_base)
    dev = np.arange(n) < int(0.2 * n)
    test = ~dev
    base_t = float(-lp_base[test].mean())
    p_base = np.exp(lp_base)
    out = {"model": which, "domain": domain, "n": int(n), "ranks": ranks,
           "eta": ETA, "base_nll_test": base_t,
           "memory_bytes": D_G * D_V * 4}

    best_r, best_dev = None, None
    for r in ranks:
        dv = float(-lp_fw[r][dev].mean())
        tv = float(-lp_fw[r][test].mean())
        out[f"fw_r{r}"] = {
            "nll_dev": dv, "dnll_test": float(base_t - tv),
            "adapter_bytes": int(MODELS[which][1] * r * 4),
            "segments": segments(lp_base, np.exp(lp_fw[r]), n)}
        print(f"  fw_r{r}: dev {dv:.4f} | test dNLL {base_t - tv:+.4f}",
              flush=True)
        if best_dev is None or dv < best_dev:
            best_dev, best_r = dv, r
    out["fw_winner_rank"] = best_r
    p_fw = np.exp(lp_fw[best_r])

    m_only = tune_memory(p_base, s_true, smax, lse, dev)
    m_fw = tune_memory(p_fw, s_true, smax, lse, dev)
    for key, b in (("memory_only", m_only), ("fw_plus_memory", m_fw)):
        tv = float(-np.log(np.maximum(b["p"][test], 1e-30)).mean())
        out[key] = {"dnll_test": float(base_t - tv), "params": b["params"],
                    "segments": segments(lp_base, b["p"], n)}
        print(f"  {key}: test dNLL {base_t - tv:+.4f} {b['params']}",
              flush=True)
    out["paired_fw_vs_base"] = paired(p_base, p_fw, test)
    out["paired_combo_vs_memory"] = paired(m_only["p"], m_fw["p"], test)
    print(f"  paired FW vs base   : {out['paired_fw_vs_base']['mean']:+.4f} "
          f"P={out['paired_fw_vs_base']['p_better']:.3f}")
    print(f"  paired combo vs mem : "
          f"{out['paired_combo_vs_memory']['mean']:+.4f} "
          f"CI {out['paired_combo_vs_memory']['ci95']} "
          f"P={out['paired_combo_vs_memory']['p_better']:.3f}")
    print(f"  FW segments: "
          f"{[round(v, 4) for v in out[f'fw_r{best_r}']['segments']]}")
    with open(f"results/fwscale_{prefix}{domain}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
