"""The state of a Sillage memory: four gradient-free mechanisms, one object.

Everything a frozen language model learns at test time lives here, and
nothing else does. No gradients are ever computed; the only signal used is
the model's own token surprise g = clip(-ln p_LM, 0, 5), which is free at
inference.

    M_G   fast n-gram Hebbian matrix (4096 x 256, 4.2 MB) -- keys are sliding
          n-gram bindings of random token hypervectors, values are amplitudes
          (square roots of accumulated mass), writes scaled by g.   [paper 1]
    M_S   fast semantic matrix (12288 x 256, 12.6 MB) -- keys are banded
          SimHash symbols over the model's hidden state, mixed at SCORE level
          with a per-block confidence threshold, never in the key. [paper 2]
    cold  bounded exact n-gram store, consolidated by surprise MASS when the
          session ends: high-value grams survive, the rest is pruned.
                                                                   [paper 3]
    A     rank-16 delta-rule readout adapter, l' = l + A phi(h), updated with
          a UNIFORM step eta * mean(g) -- surprise gating hurts this one.
                                                                   [paper 4]

An optional per-token leak (half-life in tokens, applied in blocks of 64 like
the 500k-token experiments) implements forgetting; it is off by default and
matters past ~0.5 writes per parameter.
"""

import json
import os
import pickle

import numpy as np

# --- paper 1: n-gram tier ---------------------------------------------------
D_K, D_V, NGRAM, CAP = 4096, 256, 4, 5.0
# --- paper 2: semantic tier -------------------------------------------------
L_BANDS, B_BITS, D_BAND = 32, 16, 128
B_LIST = [8, 12, 16]
D_S = len(B_LIST) * L_BANDS * D_BAND
# --- paper 3: cold store ----------------------------------------------------
COLD_MAX, COLD_MIN_COUNT, LAM_C = 50_000, 2, 0.3
# --- paper 4: readout adapter -----------------------------------------------
R_FEAT, ETA = 16, 0.1
# --- forgetting (papers 1 and 3) -------------------------------------------
DECAY_EVERY = 64

RESERVOIR = 5000
SEED_V, SEED_T, SEED_W, SEED_R = 7001, 7002, 7003, 7010

MODELS = {  # name: (hub id, vocab, (beta_G, lam_G), (beta_S, lam_S), semantic)
    "qwen": ("Qwen/Qwen3-0.6B", 151_936, (160.0, 0.2), (40.0, 0.1), True),
    "gpt2": ("openai-community/gpt2", 50_257, (40.0, 0.3), (40.0, 0.1),
             False),
}


def band_vec(band, pattern):
    """Deterministic hypervector for one (band, bit-pattern) symbol."""
    seed = (0x9E3779B97F4A7C15 * (band * 65537 + pattern + 1)) % 2 ** 64
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=D_BAND) * 2.0 - 1.0).astype(np.float32)


class SillageMemory:
    """The mechanisms, without any language model attached.

    Feed it logits + hidden states + the observed token and it does the rest;
    `sillage.Sillage` is the wrapper that owns a frozen transformer and calls
    into this. Keeping them apart is what lets the memory be tested (and
    plugged into someone else's generation loop) with no transformers import.
    """

    def __init__(self, state_dir=None, which="qwen", semantic=None,
                 fastweights=None, half_life=None):
        _, vocab, (bG, lG), (bS, lS), sem_default = MODELS[which]
        self.dir = state_dir
        self.which = which
        self.vocab = vocab
        self.beta_G, self.lam_G = bG, lG
        self.beta_S, self.lam_S = bS, lS
        self.semantic = sem_default if semantic is None else semantic
        self.fastweights = fastweights          # None -> whatever the state
        self.half_life = half_life              #         was built with
        self._sem_arg = semantic
        self._V = None                  # hypervectors are regenerated from
        self._T = None                  # seeds, never stored (and never
        self._Wh = None                 # allocated at all by `ask`/`status`)
        self._Rf = None
        self._band_cache = {}
        self._since_decay = 0
        self.load()

    # ---------------------------------------------------------------- state --
    def _blank(self):
        self.M = np.zeros((D_K, D_V), dtype=np.float32)
        self.MS = np.zeros((D_S, D_V), dtype=np.float32)
        self.A = np.zeros((self.vocab, R_FEAT), dtype=np.float32)
        self.mu, self.mu_n = None, 0
        self.res_G, self.res_S = [], []
        self.tokens = 0
        self.g_sum, self.g_cnt = 0.0, 0
        self.cold = {}
        self.log = {"files": []}

    def load(self):
        path = None if self.dir is None else os.path.join(self.dir,
                                                          "state.npz")
        if path is None or not os.path.exists(path):
            if self.fastweights is None:
                self.fastweights = True
            self._blank()
            return
        z = np.load(path, allow_pickle=False)
        assert str(z["model"]) == self.which, \
            f"this memory was built with --model {z['model']}"
        self.M = z["M"].astype(np.float32)
        self.MS = (z["MS"].astype(np.float32) if "MS" in z
                   else np.zeros((D_S, D_V), np.float32))
        # "A" arrived with the readout adapter (paper 4); "reservoir" is the
        # pre-semantic-tier name of res_G. Older states stay loadable.
        self.A = (z["A"].astype(np.float32) if "A" in z
                  else np.zeros((self.vocab, R_FEAT), np.float32))
        self.mu = z["mu"].astype(np.float32) if "mu" in z else None
        self.mu_n = int(z["mu_n"]) if "mu_n" in z else 0
        if "res_G" in z:
            self.res_G = list(z["res_G"])
        elif "reservoir" in z:
            self.res_G = list(z["reservoir"])
        else:
            self.res_G = []
        self.res_S = list(z["res_S"]) if "res_S" in z else []
        self.tokens = int(z["tokens"])
        self.g_sum = float(z["g_sum"]) if "g_sum" in z else 0.0
        self.g_cnt = int(z["g_cnt"]) if "g_cnt" in z else 0
        if "fastweights" in z and self.fastweights is None:
            self.fastweights = bool(z["fastweights"])
        if self.fastweights is None:
            self.fastweights = True
        if "semantic" in z and self._sem_arg is None:
            self.semantic = bool(z["semantic"])
        if "half_life" in z and self.half_life is None:
            hl = float(z["half_life"])
            self.half_life = hl if hl > 0 else None
        cold_path = os.path.join(self.dir, "cold.pkl")
        if os.path.exists(cold_path):
            with open(cold_path, "rb") as f:
                self.cold = pickle.load(f)
        else:
            self.cold = {}
        log_path = os.path.join(self.dir, "log.json")
        self.log = (json.load(open(log_path, encoding="utf-8"))
                    if os.path.exists(log_path) else {"files": []})

    def save(self):
        """Consolidate ("sleep") and write the state to disk."""
        if self.dir is None:
            return
        if len(self.cold) > COLD_MAX:      # keep the highest surprise mass
            keep = sorted(self.cold.items(), key=lambda kv: -kv[1][0])
            self.cold = dict(keep[:COLD_MAX])
        os.makedirs(self.dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(self.dir, "state.npz"), M=self.M, MS=self.MS,
            A=self.A, mu=(self.mu if self.mu is not None else np.zeros(1)),
            mu_n=self.mu_n,
            res_G=np.array(self.res_G[-RESERVOIR:], dtype=np.float32),
            res_S=np.array(self.res_S[-RESERVOIR:], dtype=np.float32),
            tokens=self.tokens, g_sum=self.g_sum, g_cnt=self.g_cnt,
            half_life=(self.half_life or 0.0), model=self.which,
            fastweights=bool(self.fastweights), semantic=bool(self.semantic))
        with open(os.path.join(self.dir, "cold.pkl"), "wb") as f:
            pickle.dump(self.cold, f)
        with open(os.path.join(self.dir, "log.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.log, f, indent=2)

    # ------------------------------------------------------ projections ------
    @property
    def V(self):
        """Token value hypervectors (vocab x 256), regenerated from a seed."""
        if self._V is None:
            rng = np.random.default_rng(SEED_V)
            self._V = ((rng.integers(0, 2, size=(self.vocab, D_V)) * 2.0 - 1.0)
                       / np.sqrt(D_V)).astype(np.float32)
        return self._V

    @property
    def T(self):
        """Token key hypervectors (vocab x 4096), regenerated from a seed."""
        if self._T is None:
            rng = np.random.default_rng(SEED_T)
            self._T = (rng.integers(0, 2, size=(self.vocab, D_K),
                                    dtype=np.int8) * 2 - 1)
        return self._T

    def Wh(self, hidden_dim):
        if self._Wh is None:
            rng = np.random.default_rng(SEED_W)
            self._Wh = rng.normal(
                size=(hidden_dim, L_BANDS * B_BITS)).astype(np.float32)
        return self._Wh

    def Rf(self, hidden_dim):
        if self._Rf is None:
            rng = np.random.default_rng(SEED_R)
            self._Rf = (rng.normal(size=(hidden_dim, R_FEAT))
                        / np.sqrt(hidden_dim)).astype(np.float32)
        return self._Rf

    # ------------------------------------------------------- keys -----------
    def new_stream(self):
        """Start a new text: the sliding n-gram key restarts empty."""
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

    def phi(self, h):
        """Fixed random feature of the hidden state, for the readout adapter."""
        v = h @ self.Rf(len(h))
        return v / (np.linalg.norm(v) + 1e-8)

    # ------------------------------------------------------- read ------------
    @staticmethod
    def _thr(res):
        """Confidence threshold: abstain below the dev 75th percentile."""
        if len(res) < 500:
            return np.inf
        return float(np.quantile(res, 0.75))

    def thresholds(self):
        return self._thr(self.res_G), self._thr(self.res_S)

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

    def adapt(self, logits, h):
        """Fast weights: l' = l + A phi(h). Returns (logits', phi)."""
        if not self.fastweights or h is None:
            return logits, None
        f = self.phi(h)
        return logits + self.A @ f, f

    def mix_true(self, p_base_true, sG, truth, sS=None, pc=None,
                 thrG=None, thrS=None):
        """Mixed probability of one known token (reading-time perplexity)."""
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
        """Mixed full distribution (generation time)."""
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

    # ------------------------------------------------------- write -----------
    def amp_write(self, M, q, u, tok_next, g):
        """Amplitude update: coefficients are square roots of stored mass."""
        a = max(0.0, float(u @ self.V[tok_next]))
        M += (np.sqrt(a * a + g) - a) * q[:, None] * self.V[tok_next][None, :]

    def fw_update(self, phi, p_adapted, truth):
        """Delta rule at the readout, uniform step (gating hurts it)."""
        if not self.fastweights or phi is None:
            return
        step = ETA * (self.g_sum / max(1, self.g_cnt))
        if step <= 0:
            return
        self.A -= (step * p_adapted)[:, None] * phi[None, :]
        self.A[truth] += step * phi

    def decay_step(self):
        """Leaky forgetting, applied in blocks of 64 tokens (papers 1, 3)."""
        if not self.half_life:
            return
        self._since_decay += 1
        if self._since_decay >= DECAY_EVERY:
            self._since_decay = 0
            gamma = 0.5 ** (DECAY_EVERY / float(self.half_life))
            self.M *= gamma
            if self.semantic:
                self.MS *= gamma

    def write_all(self, qG, uG, qS, uS, tok_next, g, phi=None, p_adapted=None):
        """One token: every tier writes, then the whole state ages."""
        self.amp_write(self.M, qG, uG, tok_next, g)
        if qS is not None:
            self.amp_write(self.MS, qS, uS, tok_next, g)
        if len(self._hist) >= NGRAM:
            gram = np.array(self._hist[-NGRAM:], dtype=np.int32).tobytes()
            slot = self.cold.setdefault(gram, [0.0, {}])
            slot[0] += g                      # surprise mass, for consolidation
            slot[1][int(tok_next)] = slot[1].get(int(tok_next), 0) + 1
        self.g_sum += g
        self.g_cnt += 1
        if p_adapted is not None:
            self.fw_update(phi, p_adapted, tok_next)
        self.tokens += 1
        self.decay_step()

    # ------------------------------------------------------- reporting -------
    def writes_per_parameter(self):
        return self.tokens / float(D_K * D_V)

    def sizes(self):
        """Bytes held by each tier, in the order the papers introduce them."""
        return {
            "n-gram memory M_G": self.M.nbytes,
            "semantic memory M_S": self.MS.nbytes if self.semantic else 0,
            "cold store": sum(64 + 24 * len(v[1]) for v in self.cold.values()),
            "fast weights A": self.A.nbytes if self.fastweights else 0,
        }
