"""Sillage Assistant v2 — a frozen LM that remembers what you gave it to read.

Memory = the paper-3 hierarchy:
  * fast n-gram Hebbian matrix M_G (4.2 MB, amplitude writes, surprise gate)
  * fast SEMANTIC matrix M_S (banded SimHash over hidden states, 12.6 MB) —
    on by default for qwen (well-conditioned raw geometry; paper-2 rule),
    off for gpt2 (would need whitening); toggle with --semantic/--no-semantic
  * bounded cold store of exact 4-grams, consolidated BY SURPRISE MASS when
    the session ends ("sleep": high-value grams persist, the rest is pruned)
State persists across sessions in ./memory_state/ (a few MB on disk).
The assistant never learns from its own generations — only from reading.

Commands
  python assistant.py read <file> [<file> ...]      read + memorize
  python assistant.py complete "prompt" [-n N] [--temp T]
  python assistant.py status
  python assistant.py forget --all

Options: --model qwen|gpt2 (default qwen = Qwen3-0.6B), --state DIR.
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

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

D_K, D_V, NGRAM, CAP = 4096, 256, 4, 5.0
L_BANDS, B_BITS, D_BAND = 32, 16, 128
B_LIST = [8, 12, 16]
D_S = len(B_LIST) * L_BANDS * D_BAND
WINDOW, STRIDE = 1024, 512
COLD_MAX = 50_000
COLD_MIN_COUNT = 2
LAM_C = 0.3
RESERVOIR = 5000
MODELS = {  # name, vocab, (beta_G, lam_G), (beta_S, lam_S), semantic default
    "qwen": ("Qwen/Qwen3-0.6B", 151_936, (160.0, 0.2), (40.0, 0.1), True),
    "gpt2": ("openai-community/gpt2", 50_257, (40.0, 0.3), (40.0, 0.1),
             False),
}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def band_vec(band, pattern):
    seed = (0x9E3779B97F4A7C15 * (band * 65537 + pattern + 1)) % 2 ** 64
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=D_BAND) * 2.0 - 1.0).astype(np.float32)


class Memory:
    def __init__(self, state_dir, which, semantic=None):
        self.dir = state_dir
        self.which = which
        name, vocab, (bG, lG), (bS, lS), sem_default = MODELS[which]
        self.vocab = vocab
        self.beta_G, self.lam_G = bG, lG
        self.beta_S, self.lam_S = bS, lS
        self.semantic = sem_default if semantic is None else semantic
        rngV = np.random.default_rng(7001)
        rngT = np.random.default_rng(7002)
        self.V = ((rngV.integers(0, 2, size=(vocab, D_V)) * 2.0 - 1.0)
                  / np.sqrt(D_V)).astype(np.float32)
        self.T = (rngT.integers(0, 2, size=(vocab, D_K), dtype=np.int8)
                  * 2 - 1)
        self._Wh = None
        self._band_cache = {}
        p = os.path.join(state_dir, "state.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=False)
            assert str(z["model"]) == which, \
                f"state was built with --model {z['model']}"
            self.M = z["M"].astype(np.float32)
            self.MS = z["MS"].astype(np.float32) if "MS" in z else \
                np.zeros((D_S, D_V), np.float32)
            self.mu = z["mu"].astype(np.float32) if "mu" in z else None
            self.mu_n = int(z["mu_n"]) if "mu_n" in z else 0
            self.res_G = list(z["res_G"])
            self.res_S = list(z["res_S"]) if "res_S" in z else []
            self.tokens = int(z["tokens"])
            with open(os.path.join(state_dir, "cold.pkl"), "rb") as f:
                self.cold = pickle.load(f)
            self.log = json.load(open(os.path.join(state_dir, "log.json")))
        else:
            self.M = np.zeros((D_K, D_V), dtype=np.float32)
            self.MS = np.zeros((D_S, D_V), dtype=np.float32)
            self.mu, self.mu_n = None, 0
            self.res_G, self.res_S = [], []
            self.tokens = 0
            self.cold = {}
            self.log = {"files": []}

    def Wh(self, hidden_dim):
        if self._Wh is None:
            rngW = np.random.default_rng(7003)
            self._Wh = rngW.normal(
                size=(hidden_dim, L_BANDS * B_BITS)).astype(np.float32)
        return self._Wh

    # -- streaming primitives -------------------------------------------------
    def new_stream(self):
        self._graw = np.ones(D_K, dtype=np.float32)
        self._hist = []

    def step_key(self, tok):
        self._graw = np.roll(self._graw, 1)
        self._graw *= self.T[tok]
        self._hist.append(int(tok))
        if len(self._hist) > NGRAM:
            old = self._hist.pop(0)
            self._graw *= np.roll(self.T[old], NGRAM)
        return self._graw / np.sqrt(D_K)

    def sem_key(self, h):
        h = h / (np.linalg.norm(h) + 1e-8)
        if self.mu is None:
            self.mu = np.zeros_like(h)
        self.mu_n += 1
        self.mu += (h - self.mu) / self.mu_n
        z = h - self.mu
        bits = ((z @ self.Wh(len(h))) > 0).reshape(L_BANDS, B_BITS)
        q = np.empty(D_S, dtype=np.float32)
        scale = 1.0 / np.sqrt(len(B_LIST) * L_BANDS * D_BAND)
        pw2 = 2 ** np.arange(B_BITS)
        slot = 0
        for gi, b in enumerate(B_LIST):
            for k in range(L_BANDS):
                pat = int(bits[k, :b] @ pw2[:b])
                key = (gi * L_BANDS + k, pat)
                v = self._band_cache.get(key)
                if v is None:
                    v = band_vec(*key)
                    self._band_cache[key] = v
                q[slot * D_BAND:(slot + 1) * D_BAND] = scale * v
                slot += 1
        return q

    @staticmethod
    def _thr(res):
        if len(res) < 500:
            return np.inf
        return float(np.quantile(res, 0.75))

    def scores(self, M, q):
        u = M.T @ q
        un = float(np.linalg.norm(u)) + 1e-8
        return u, (self.V @ u) / un

    def cold_lookup(self, tok_next=None):
        if len(self._hist) < NGRAM:
            return None
        gram = np.array(self._hist[-NGRAM:], dtype=np.int32).tobytes()
        slot = self.cold.get(gram)
        if slot is None or sum(slot[1].values()) < COLD_MIN_COUNT:
            return None
        tot = sum(slot[1].values())
        if tok_next is not None:
            return slot[1].get(int(tok_next), 0) / tot
        return {t: c / tot for t, c in slot[1].items()}

    def amp_write(self, M, q, u, tok_next, g):
        a = max(0.0, float(u @ self.V[tok_next]))
        M += (np.sqrt(a * a + g) - a) * q[:, None] * self.V[tok_next][None, :]

    def write_all(self, qG, uG, qS, uS, tok_next, g):
        self.amp_write(self.M, qG, uG, tok_next, g)
        if qS is not None:
            self.amp_write(self.MS, qS, uS, tok_next, g)
        if len(self._hist) >= NGRAM:
            gram = np.array(self._hist[-NGRAM:], dtype=np.int32).tobytes()
            slot = self.cold.setdefault(gram, [0.0, {}])
            slot[0] += g
            slot[1][int(tok_next)] = slot[1].get(int(tok_next), 0) + 1
        self.tokens += 1

    def mix_true(self, p_base_true, sG, truth, sS=None, pc=None,
                 thrG=None, thrS=None):
        """Mixed probability of one known token (reading-time PPL)."""
        p = p_base_true
        if float(sG.max()) >= (thrG if thrG is not None else np.inf):
            m = self.beta_G * sG
            mx = m.max()
            pm = float(np.exp(self.beta_G * sG[truth]
                              - (mx + np.log(np.exp(m - mx).sum()))))
            p = self.lam_G * pm + (1 - self.lam_G) * p
        if sS is not None and \
                float(sS.max()) >= (thrS if thrS is not None else np.inf):
            m = self.beta_S * sS
            mx = m.max()
            pm = float(np.exp(self.beta_S * sS[truth]
                              - (mx + np.log(np.exp(m - mx).sum()))))
            p = self.lam_S * pm + (1 - self.lam_S) * p
        if pc is not None:
            p = LAM_C * pc + (1 - LAM_C) * p
        return p

    def mix_full(self, p_base, sG, sS=None, pc=None, thrG=None, thrS=None):
        """Mixed full distribution (generation-time)."""
        p = p_base
        for s, beta, lam, thr in ((sG, self.beta_G, self.lam_G, thrG),
                                  (sS, self.beta_S, self.lam_S, thrS)):
            if s is None:
                continue
            if float(s.max()) >= (thr if thr is not None else np.inf):
                m = beta * s
                pm = np.exp(m - m.max())
                pm /= pm.sum()
                p = lam * pm + (1 - lam) * p
        if pc is not None:
            p = (1 - LAM_C) * p
            for t, pv in pc.items():
                p[t] += LAM_C * pv
        return p

    def sleep_and_save(self):
        if len(self.cold) > COLD_MAX:
            keep = sorted(self.cold.items(), key=lambda kv: -kv[1][0])
            self.cold = dict(keep[:COLD_MAX])
        os.makedirs(self.dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.dir, "state.npz"), M=self.M, MS=self.MS,
            mu=(self.mu if self.mu is not None else np.zeros(1)),
            mu_n=self.mu_n,
            res_G=np.array(self.res_G[-RESERVOIR:], dtype=np.float32),
            res_S=np.array(self.res_S[-RESERVOIR:], dtype=np.float32),
            tokens=self.tokens, model=self.which)
        with open(os.path.join(self.dir, "cold.pkl"), "wb") as f:
            pickle.dump(self.cold, f)
        with open(os.path.join(self.dir, "log.json"), "w") as f:
            json.dump(self.log, f, indent=2)


def load_model(which):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[which][0]
    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    return tok, model


@torch.no_grad()
def cmd_read(mem, tok, model, paths):
    for path in paths:
        base = os.path.basename(path)
        if any(f["file"] == base for f in mem.log["files"]):
            print(f"note: {base} was read before -- re-reading strengthens "
                  f"its traces.")
        text = open(path, encoding="utf-8", errors="replace").read()
        ids = np.array(tok.encode(text), dtype=np.int64)
        n = len(ids) - 1
        mem.new_stream()
        thrG, thrS = mem._thr(mem.res_G), mem._thr(mem.res_S)
        nll_b = nll_m = 0.0
        cnt = 0
        x = torch.tensor(ids)
        a = 0
        t0 = time.time()
        while a < n:
            w = min(WINDOW, len(ids) - a)
            out = model(x[a:a + w].unsqueeze(0),
                        output_hidden_states=mem.semantic)
            logprobs = torch.log_softmax(out.logits[0].float(), -1).numpy()
            hs = out.hidden_states[-1][0].float().numpy() if mem.semantic \
                else None
            lo = 0 if a == 0 else WINDOW - STRIDE
            for i in range(lo, w):
                j = a + i
                if j >= n:
                    break
                truth = int(ids[j + 1])
                lp = float(logprobs[i, truth])
                qG = mem.step_key(int(ids[j]))
                uG, sG = mem.scores(mem.M, qG)
                mem.res_G.append(float(sG.max()))
                qS = uS = sS = None
                if mem.semantic:
                    qS = mem.sem_key(hs[i])
                    uS, sS = mem.scores(mem.MS, qS)
                    mem.res_S.append(float(sS.max()))
                pc = mem.cold_lookup(truth)
                p = mem.mix_true(np.exp(lp), sG, truth, sS, pc, thrG, thrS)
                nll_b += -lp
                nll_m += -np.log(max(p, 1e-30))
                cnt += 1
                g = min(CAP, max(0.0, -lp))
                mem.write_all(qG, uG, qS, uS, truth, g)
            if a + w >= len(ids):
                break
            a += STRIDE
        ppl_b, ppl_m = np.exp(nll_b / cnt), np.exp(nll_m / cnt)
        mem.res_G = mem.res_G[-RESERVOIR:]
        mem.res_S = mem.res_S[-RESERVOIR:]
        mem.log["files"].append({
            "file": base, "tokens": int(n),
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "ppl_frozen": round(float(ppl_b), 2),
            "ppl_with_memory": round(float(ppl_m), 2)})
        print(f"read {base}: {n} tokens in {(time.time()-t0)/60:.1f} min | "
              f"PPL frozen {ppl_b:.2f} -> with memory {ppl_m:.2f}",
              flush=True)
    mem.sleep_and_save()
    print(f"memory consolidated and saved ({mem.tokens} tokens lifetime, "
          f"{len(mem.cold)} cold grams).")


@torch.no_grad()
def cmd_complete(mem, tok, model, prompt, n_tokens, temp):
    ids = tok.encode(prompt)
    mem.new_stream()
    for t in ids[:-1]:
        mem.step_key(int(t))
    thrG, thrS = mem._thr(mem.res_G), mem._thr(mem.res_S)
    rng = np.random.default_rng(0)
    past = None
    inp = torch.tensor(ids).unsqueeze(0)
    out_ids = []
    for _ in range(n_tokens):
        out = model(inp, past_key_values=past, use_cache=True,
                    output_hidden_states=mem.semantic)
        past = out.past_key_values
        p_base = torch.softmax(out.logits[0, -1].float(), -1).numpy()
        qG = mem.step_key(int(inp[0, -1]))
        _, sG = mem.scores(mem.M, qG)
        sS = None
        if mem.semantic:
            h = out.hidden_states[-1][0, -1].float().numpy()
            qS = mem.sem_key(h)
            _, sS = mem.scores(mem.MS, qS)
        pc = mem.cold_lookup()
        p = mem.mix_full(p_base, sG, sS, pc, thrG, thrS)
        if temp and temp > 0:
            logp = np.log(np.maximum(p, 1e-30)) / temp
            pp = np.exp(logp - logp.max())
            pp /= pp.sum()
            nxt = int(rng.choice(len(pp), p=pp))
        else:
            nxt = int(np.argmax(p))
        out_ids.append(nxt)
        inp = torch.tensor([[nxt]])
        if nxt == getattr(tok, "eos_token_id", -1):
            break
    print(prompt + tok.decode(out_ids))


def cmd_status(mem):
    size = sum(os.path.getsize(os.path.join(mem.dir, f))
               for f in ("state.npz", "cold.pkl")
               if os.path.exists(os.path.join(mem.dir, f)))
    print(f"model: {MODELS[mem.which][0]} "
          f"(semantic tier {'on' if mem.semantic else 'off'})")
    print(f"lifetime tokens read : {mem.tokens}")
    print(f"cold grams stored    : {len(mem.cold)}")
    print(f"state on disk        : {size/1e6:.1f} MB ({mem.dir})")
    for f in mem.log["files"][-10:]:
        print(f"  {f['date']}  {f['file']}: {f['tokens']} tok, "
              f"PPL {f['ppl_frozen']} -> {f['ppl_with_memory']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["read", "complete", "status", "forget"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--state", default="memory_state")
    ap.add_argument("-n", "--tokens", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--semantic", dest="semantic", action="store_true",
                    default=None)
    ap.add_argument("--no-semantic", dest="semantic", action="store_false")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.cmd == "forget":
        if a.all:
            for f in ("state.npz", "cold.pkl", "log.json"):
                p = os.path.join(a.state, f)
                if os.path.exists(p):
                    os.remove(p)
            print("memory wiped.")
        else:
            print("use: forget --all")
        return
    mem = Memory(a.state, a.model, a.semantic)
    if a.cmd == "status":
        cmd_status(mem)
        return
    tok, model = load_model(a.model)
    if a.cmd == "read":
        cmd_read(mem, tok, model, a.args)
    elif a.cmd == "complete":
        cmd_complete(mem, tok, model, a.args[0], a.tokens, a.temp)


if __name__ == "__main__":
    main()
