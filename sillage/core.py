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

How loudly each tier speaks is not hard-coded for an unfamiliar model: a
rolling window of what was just read becomes a development split, the
published grids of (beta, lambda, abstention threshold) are searched on it,
and the winner governs the next read -- the tuning protocol of the papers,
run inside the tool. That is what lets the same code sit behind any frozen
model. For the two models the papers DID tune, their published settings are
kept instead, because a window read by a cold memory measurably loses to
them (the numbers are next to BETAS below).
"""

import json
import os
import sys

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

MODELS = {  # shortcut: (hub id, vocab, (beta_G, lam_G), (beta_S, lam_S), sem)
    "qwen": ("Qwen/Qwen3-0.6B", 151_936, (160.0, 0.2), (40.0, 0.1), True),
    "gpt2": ("openai-community/gpt2", 50_257, (40.0, 0.3), (40.0, 0.1),
             False),
}
TUNED = set(MODELS)          # models the papers measured with a full protocol
# Paper 8: whether the v2 tier needs its keys whitened is a property
# of the model's geometry, and no cheap proxy we tried predicts it
# (see runtime.resolve_sem2). These two were measured; anything else
# whitens by default, which is paper 2's rule.
SEM2_WHITEN = {"qwen": False, "gpt2": True}
# Same rule as the readout settings above: what the papers measured wins
# for the models they measured. GPT-2's identity profile is nearly flat
# (0.47-0.52 across its lower half), so a sweep cannot tell layer 5 from
# layer 6 -- and layer 6 does not work. `--sem2 auto` therefore uses
# these for known models and sweeps only for the rest.
SEM2_LAYER = {"qwen": 1, "gpt2": 5}
# Any other causal LM works too. It starts from the GPT-2 settings -- a sane
# middle of the grid -- and then calibrates them on what it reads (see
# maybe_calibrate). The semantic tier stays OFF for an unknown model: paper 2
# shows raw hidden states need whitening except where their geometry is
# already well conditioned (Qwen3), so it is opt-in with --semantic.
DEFAULT_G, DEFAULT_S = (40.0, 0.3), (40.0, 0.1)

# --- readout calibration (the tuning protocol of the papers, run online) ---
# Same grids as memory/memories.py, plus the beta = 320 the appendix of paper
# 1 checked at the grid edge: a new model may well want a sharper readout
# than either of the two studied ones.
#
# Calibration is ON for an unknown model and OFF for the two shortcuts, and
# that is a measured decision, not a preference. Fitting on your own stream
# means fitting on a COLD memory -- the window is read before the memory has
# absorbed it -- while the settings then govern a warmer one. Measured on
# GPT-2 over two technical papers (fit on the first, scored on the second):
# whole window +0.098 nats, recent half +0.103, recent quarter +0.109,
# shipped defaults +0.120, oracle (fitted on the second paper itself) +0.125.
# Where someone has already tuned a model on a long stream, that tuning wins;
# where nobody has, a cold fit is still the best estimate available.
BETAS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
LAMS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85)
THRESH_Q = (None, 0.25, 0.5, 0.75)     # None = never abstain
CALIB_MIN, CALIB_MAX = 600, 4000       # observations needed / kept
CALIB_EVERY = 3                        # sample one position in three
DEFAULT_THR_Q = 0.75
# Paper 8's tier, as measured IN THE TOOL (the paper's own numbers come
# from prototypes with dense keys; banded keys pack their scores more
# tightly, so beta must be larger here -- the pooled max sits ~0.5 with
# a 0.09 median, and a token needs beta*(s_max - s_typical) > ln(vocab)
# to carry the argmax). Grid on ten development facts, reported on ten
# held-out ones; calibration still overrides all three.
SEM2_BETA_S, SEM2_LAM_S = 20.0, 0.85
SEM2_THR_Q = 0.90                      # abstain below the q90 of the
                                       # NULL: ordinary positions of
                                       # the document, sampled before
                                       # the tier's own writes weigh in


def resolve(which, state_dir=None):
    """Settings for a model name, a shortcut, or a local path.

    Order of trust: the state on disk (it knows the vocabulary it was built
    with, and asking costs no network), then the shortcut table, then the
    model's own config.json.
    """
    if which in MODELS:
        return MODELS[which]
    saved = peek(state_dir)
    if saved and saved[0] == which and saved[1]:
        return (which, saved[1], DEFAULT_G, DEFAULT_S, False)
    from transformers import AutoConfig            # only config.json
    try:
        cfg = AutoConfig.from_pretrained(which)
    except Exception as exc:
        raise SystemExit(
            f"unknown model {which!r} ({type(exc).__name__}). Give a "
            f"Hugging Face id (e.g. HuggingFaceTB/SmolLM2-135M), a local "
            f"folder, or one of the shortcuts: "
            f"{', '.join(MODELS)}.")
    vocab = getattr(cfg, "vocab_size", None)
    if not vocab:
        raise SystemExit(f"cannot read vocab_size from {which}'s config -- "
                         f"is it a causal language model?")
    arch = list(getattr(cfg, "architectures", None) or [])
    if arch and not any(a.endswith(("ForCausalLM", "LMHeadModel"))
                        for a in arch):
        # transformers will happily bolt a RANDOM head onto an encoder and
        # return nonsense (perplexity in the tens of thousands). Say so.
        print(f"warning: {which} is a {arch[0]}, not a next-token predictor. "
              f"Its language-modelling head would be randomly initialised "
              f"and every number below would be meaningless.")
    return (which, int(vocab), DEFAULT_G, DEFAULT_S, False)


def peek(state_dir):
    """(model id, vocab) of an existing state, without loading it."""
    if not state_dir:
        return None
    path = os.path.join(state_dir, "state.npz")
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return (str(z["model"]),
                    int(z["vocab"]) if "vocab" in z else 0)
    except Exception:
        return None


def lse_grid(s):
    """log-sum-exp of beta*s for every beta in the grid (one pass each)."""
    out = np.empty(len(BETAS), dtype=np.float32)
    for i, b in enumerate(BETAS):
        m = b * s
        mx = float(m.max())
        out[i] = mx + np.log(np.exp(m - mx).sum())
    return out


def fit_readout(p_basis, s_true, s_max, lse):
    """Pick (beta, lambda, threshold) that minimise dev NLL, as in the papers.

    `p_basis` is the probability the true token already has before this tier
    speaks (the frozen model, adapted by the fast weights, and for the
    semantic tier also mixed with the n-gram tier). Returns the winner and
    the probabilities it produces, so the next tier can be fitted on top.
    """
    best = None
    for bi, beta in enumerate(BETAS):
        p_mem = np.exp(beta * s_true - lse[:, bi])
        for q in THRESH_Q:
            if q is None:
                mask = np.ones(len(p_basis), dtype=bool)
            else:
                mask = s_max >= np.quantile(s_max, q)
            for lam in LAMS:
                p = np.where(mask, lam * p_mem + (1 - lam) * p_basis, p_basis)
                nll = float(-np.log(np.maximum(p, 1e-30)).mean())
                if best is None or nll < best[0]:
                    best = (nll, float(beta), float(lam), q, p)
    return best


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

    #: tiers that spoke at the last ``mix_full`` -- see that method
    last_src = ()

    def __init__(self, state_dir=None, which=None, semantic=None,
                 fastweights=None, half_life=None, calibrate=None,
                 cold_mass=None, sem2=None, sem2_whiten=None):
        self.dir = state_dir
        if which is None:              # adopt whatever this state belongs to
            saved = peek(state_dir)
            which = saved[0] if saved else "qwen"
        hub, vocab, (bG, lG), (bS, lS), sem_default = resolve(which, state_dir)
        self.which = which
        self.hub = hub
        self.vocab = vocab
        self.beta_G, self.lam_G = bG, lG
        self.beta_S, self.lam_S = bS, lS
        self.thr_qG = self.thr_qS = DEFAULT_THR_Q
        # calibrate an unfamiliar model; trust the papers for the two they
        # measured, unless asked otherwise
        self.calibrate_on = (bool(calibrate) if calibrate is not None
                             else which not in TUNED)
        self.calibrated = False
        self.cal_at = 0                 # lifetime tokens at the last fit
        self.cal = None                 # dev statistics, until they are used
        self.semantic = sem_default if semantic is None else semantic
        # "auto" defers the layer choice to the first read: the layer
        # paper 8 measured for a model it measured, otherwise a sweep
        # over the document itself (runtime.resolve_sem2, which scores
        # candidate layers with sem2_separation)
        self.sem2_auto = (sem2 == "auto")
        if self.sem2_auto:
            sem2 = None
        if sem2 is not None or self.sem2_auto:   # v2 keys need the tier on
            self.semantic = True
            # the published S-readout was tuned for the old tier's
            # likelihood smoothing; the v2 tier addresses, and carries
            # its measured mixing (the --target precedent)
            self.beta_S, self.lam_S = SEM2_BETA_S, SEM2_LAM_S
            self.thr_qS = SEM2_THR_Q
        self._sem2_arg = (sem2 if not self.sem2_auto else "auto")
        self._sem2_whiten_arg = sem2_whiten
        self.sem2_layer = sem2
        self.sem2_whiten = bool(sem2_whiten) if sem2_whiten is not None \
            else False
        self.fastweights = fastweights          # None -> whatever the state
        self.half_life = half_life              #         was built with
        # cold-store successor weighting: counts (historical, the papers'
        # numbers) or surprise mass per successor (paper 6's fix -- the one
        # gate-blind quantity in the system was the attack channel). Opt-in.
        self.cold_mass = cold_mass
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
        self.mu2, self.mu2_n = None, 0
        self.cov_S2, self._W2 = None, None
        self.res_G, self.res_S = [], []
        self.tokens = 0
        self.g_sum, self.g_cnt = 0.0, 0
        self.cold = {}
        self.log = {"files": []}

    # ------------------------------------------------- state, safely -------
    # A state is data, not code: every part of it loads with
    # allow_pickle=False. Before 1.5 the cold store and the calibration
    # window were Python pickles, which execute code when opened -- so a
    # downloaded state could run anything. They are flat arrays now.

    @staticmethod
    def _cold_save(cold, path):
        """The cold store as CSR arrays (grams, then successors)."""
        n = len(cold)
        grams = np.zeros((n, NGRAM), np.int32)
        mass = np.zeros(n, np.float32)
        offs = np.zeros(n + 1, np.int64)
        toks, cnts, smass = [], [], []
        for i, (gram, slot) in enumerate(cold.items()):
            if len(gram) != NGRAM * 4:
                raise ValueError(
                    f"cold key of {len(gram)} bytes: this store holds "
                    f"{NGRAM}-gram keys ({NGRAM * 4} bytes) and nothing "
                    f"else")
            grams[i] = np.frombuffer(gram, dtype=np.int32)
            mass[i] = slot[0]
            per = slot[2] if len(slot) > 2 else {}
            for t, c in slot[1].items():
                toks.append(int(t))
                cnts.append(int(c))
                smass.append(float(per.get(t, 0.0)))
            offs[i + 1] = len(toks)
        np.savez_compressed(path, grams=grams, mass=mass, offs=offs,
                            toks=np.array(toks, np.int32),
                            cnts=np.array(cnts, np.int32),
                            smass=np.array(smass, np.float32))

    @staticmethod
    def _cold_load(path):
        with np.load(path, allow_pickle=False) as z:
            grams, mass, offs = z["grams"], z["mass"], z["offs"]
            toks, cnts, smass = z["toks"], z["cnts"], z["smass"]
        cold = {}
        for i in range(len(grams)):
            a, b = int(offs[i]), int(offs[i + 1])
            cold[np.ascontiguousarray(grams[i]).tobytes()] = [
                float(mass[i]),
                {int(toks[j]): int(cnts[j]) for j in range(a, b)},
                {int(toks[j]): float(smass[j]) for j in range(a, b)}]
        return cold

    @staticmethod
    def _cal_save(cal, path):
        d = {"sem": np.array(bool(cal.get("sem", False)))}
        for k in ("p", "gt", "gm", "st", "sm"):
            d[k] = np.array(cal.get(k, []), np.float32)
        for k in ("gl", "sl"):
            rows = cal.get(k, [])
            d[k] = (np.array(rows, np.float32) if rows
                    else np.zeros((0, len(BETAS)), np.float32))
        np.savez_compressed(path, **d)

    @staticmethod
    def _cal_load(path):
        with np.load(path, allow_pickle=False) as z:
            cal = {"sem": bool(z["sem"])}
            for k in ("p", "gt", "gm", "st", "sm"):
                cal[k] = [float(x) for x in z[k]]
            for k in ("gl", "sl"):
                cal[k] = [row.copy() for row in z[k]]
        return cal

    @staticmethod
    def _unpickle_once(path, what):
        """Read one pre-1.5 part so it can be rewritten safely. This is
        the only place the tool ever unpickles, it happens once per
        state, and it says so: unpickling runs code, so migrate only
        states you wrote yourself."""
        if os.environ.get("SILLAGE_NO_PICKLE"):
            raise SystemExit(
                f"{path} is a pre-1.5 pickle and SILLAGE_NO_PICKLE is "
                f"set. Delete it, or unset the variable to migrate it.")
        print(f"note: migrating {what} from the pre-1.5 pickle format "
              f"({os.path.basename(path)}). Unpickling executes code -- "
              f"only migrate states you created yourself.",
              file=sys.stderr, flush=True)
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def load(self):
        """Read the state back, tolerating states from earlier versions."""
        path = None if self.dir is None else os.path.join(self.dir,
                                                          "state.npz")
        if path is None or not os.path.exists(path):
            if self.fastweights is None:
                self.fastweights = True
            if self.cold_mass is None:
                self.cold_mass = False
            self._blank()
            return
        z = np.load(path, allow_pickle=False)
        if str(z["model"]) != self.which:
            raise SystemExit(
                f"this memory belongs to --model {z['model']}. Use that "
                f"model, or give the new one its own --state directory: "
                f"a memory is written in one model's token space and cannot "
                f"be read in another's.")
        self.M = z["M"].astype(np.float32)
        self.MS = (z["MS"].astype(np.float32) if "MS" in z
                   else np.zeros((D_S, D_V), np.float32))
        # "A" arrived with the readout adapter (paper 4); "reservoir" is the
        # pre-semantic-tier name of res_G. Older states stay loadable.
        self.A = (z["A"].astype(np.float32) if "A" in z
                  else np.zeros((self.vocab, R_FEAT), np.float32))
        self.mu = z["mu"].astype(np.float32) if "mu" in z else None
        self.mu_n = int(z["mu_n"]) if "mu_n" in z else 0
        self.mu2 = (z["mu2"].astype(np.float32)
                    if "mu2" in z and z["mu2"].size > 1 else None)
        self.mu2_n = int(z["mu2_n"]) if "mu2_n" in z else 0
        self.cov_S2 = (z["cov_S2"].astype(np.float32)
                       if "cov_S2" in z and z["cov_S2"].size > 1 else None)
        self._W2 = None
        if "sem2_layer" in z and self._sem2_arg is None:
            v = int(z["sem2_layer"])
            self.sem2_layer = None if v < 0 else v
            if v >= 0 and self._sem_arg is not False:
                self.semantic = True   # --no-semantic still wins
                if not self.calibrated:
                    self.beta_S, self.lam_S = SEM2_BETA_S, SEM2_LAM_S
                    self.thr_qS = SEM2_THR_Q
        if "sem2_whiten" in z and self._sem2_whiten_arg is None:
            self.sem2_whiten = bool(z["sem2_whiten"])
        if "sem2_auto" in z and not self.sem2_auto:
            self.sem2_auto = bool(z["sem2_auto"])
        if self.sem2_auto and self.sem2_layer is None:
            self.semantic = self._sem_arg is not False
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
        if "cold_mass" in z and self.cold_mass is None:
            self.cold_mass = bool(z["cold_mass"])
        if bool(z["calibrated"]) if "calibrated" in z else False:
            # a calibrated readout overrides the shipped defaults: it was
            # fitted on this model, on these documents
            self.calibrated = True
            self.beta_G, self.lam_G = float(z["beta_G"]), float(z["lam_G"])
            self.beta_S, self.lam_S = float(z["beta_S"]), float(z["lam_S"])
            qg, qs = float(z["thr_qG"]), float(z["thr_qS"])
            self.thr_qG = None if qg < 0 else qg
            self.thr_qS = None if qs < 0 else qs
            self.cal_at = int(z["cal_at"]) if "cal_at" in z else self.tokens
        if self.cold_mass is None:
            self.cold_mass = False
        cal_npz = os.path.join(self.dir, "calib.npz")
        cal_old = os.path.join(self.dir, "calib.pkl")
        if self.calibrate_on and os.path.exists(cal_npz):
            self.cal = self._cal_load(cal_npz)   # the rolling window
        elif self.calibrate_on and os.path.exists(cal_old):
            self.cal = self._unpickle_once(cal_old, "the readout window")
        cold_npz = os.path.join(self.dir, "cold.npz")
        cold_old = os.path.join(self.dir, "cold.pkl")
        if os.path.exists(cold_npz):
            self.cold = self._cold_load(cold_npz)
        elif os.path.exists(cold_old):
            self.cold = self._unpickle_once(cold_old, "the cold store")
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
            vocab=self.vocab, fastweights=bool(self.fastweights),
            semantic=bool(self.semantic), cold_mass=bool(self.cold_mass),
            mu2=(self.mu2 if self.mu2 is not None else np.zeros(1)),
            mu2_n=self.mu2_n,
            cov_S2=(self.cov_S2 if self.cov_S2 is not None
                    else np.zeros((1, 1), np.float32)),
            sem2_layer=(-1 if self.sem2_layer is None
                        else int(self.sem2_layer)),
            sem2_whiten=bool(self.sem2_whiten),
            sem2_auto=bool(self.sem2_auto),
            calibrated=bool(self.calibrated),
            beta_G=self.beta_G, lam_G=self.lam_G,
            beta_S=self.beta_S, lam_S=self.lam_S,
            thr_qG=(-1.0 if self.thr_qG is None else self.thr_qG),
            thr_qS=(-1.0 if self.thr_qS is None else self.thr_qS),
            cal_at=self.cal_at)
        self._cold_save(self.cold, os.path.join(self.dir, "cold.npz"))
        with open(os.path.join(self.dir, "log.json"), "w",
                  encoding="utf-8") as f:
            json.dump(self.log, f, indent=2)
        cal_path = os.path.join(self.dir, "calib.npz")
        if self.cal and self.calibrate_on:
            self._cal_save(self.cal, cal_path)   # the rolling window
        elif os.path.exists(cal_path):
            os.remove(cal_path)
        # a migrated state keeps no pickle behind it
        for old in ("cold.pkl", "calib.pkl"):
            p = os.path.join(self.dir, old)
            if os.path.exists(p):
                os.remove(p)

    def set_vocab(self, n):
        """Adopt the model's true output width.

        Most configs report it correctly, but some pad or resize the readout,
        and the memory's value hypervectors must match the logits exactly.
        Safe while the memory is still empty; after that the traces are
        written in the old token space and cannot be reinterpreted.
        """
        if n == self.vocab:
            return
        if self.tokens:
            raise SystemExit(
                f"this memory was built for a {self.vocab}-token vocabulary "
                f"but the model outputs {n}. Start a new --state directory.")
        self.vocab = int(n)
        self._V = self._T = None
        self.A = np.zeros((self.vocab, R_FEAT), dtype=np.float32)

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
        """Random projection of the hidden state into SimHash bits."""
        if self._Wh is None:
            rng = np.random.default_rng(SEED_W)
            self._Wh = rng.normal(
                size=(hidden_dim, L_BANDS * B_BITS)).astype(np.float32)
        return self._Wh

    def Rf(self, hidden_dim):
        """Random features of the hidden state, for the readout adapter."""
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
        """Slide the n-gram key one token: bind the new, unbind the oldest."""
        self._graw = np.roll(self._graw, 1)
        self._graw *= self.T[tok]
        self._hist.append(int(tok))
        if len(self._hist) > NGRAM:
            old = self._hist.pop(0)
            self._graw *= np.roll(self.T[old], NGRAM)
        return self._graw / np.sqrt(D_K)

    def _bands(self, z):
        """Banded SimHash symbols of an already-centred vector."""
        bits = ((z @ self.Wh(len(z))) > 0).reshape(L_BANDS, B_BITS)
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

    def sem_key(self, h, learn=True):
        """Banded SimHash symbols of a centred hidden state (paper 2).

        `learn` folds this state into the running centre. It belongs at
        READ time, where the centre is part of what the memory learns,
        and nowhere else: generation is not reading, and a centre that
        drifts while answering makes the answer depend on what the
        model happened to look at last rather than on what was stored.
        Paper 8's tier already had this discipline (`sem2_observe` says
        "read time only"); this one did not, and `complete` -- whose
        own docstring promises it writes nothing -- was moving `mu` on
        every generated token.

        Measured on a 413-token French report read twice: with a clean
        centre the eight facts come back 8/8; after 182 tokens of
        unrelated prose have been folded in, 7/8, and the loss is a
        proper noun the cold store holds in full (`Brindas Kolvec`
        becomes `Brigitte Lefevre`). The tier is what carries that
        recall -- with the semantic tier off it is 7/8 either way -- so
        a drifting centre does not merely add noise, it can undo a
        recall the store had right.
        """
        h = h / (np.linalg.norm(h) + 1e-8)
        if self.mu is None:
            self.mu = np.zeros_like(h)
        if learn:
            self.mu_n += 1
            self.mu += (h - self.mu) / self.mu_n
        return self._bands(h - self.mu)

    # ------------------------------------------- paper 8: early-layer keys --
    def sem2_observe(self, h):
        """Feed one early-layer hidden into the v2 key statistics: the
        running mean, and -- when whitening is on -- the covariance the
        ZCA is estimated from. Called at read time only."""
        x = h / (np.linalg.norm(h) + 1e-8)
        if self.mu2 is None:
            self.mu2 = np.zeros_like(x)
        if self.sem2_whiten and self.cov_S2 is None:
            self.cov_S2 = np.zeros((len(x), len(x)), np.float32)
        self.mu2_n += 1
        self.mu2 += (x - self.mu2) / self.mu2_n
        if self.sem2_whiten:
            self.cov_S2 += np.outer(x, x)
        self._W2 = None

    def _W2mat(self):
        """ZCA transform from the accumulated statistics (paper 8:
        whitening is the model-adaptive component -- unnecessary where
        the layer's geometry is well conditioned, indispensable where
        it is not). Shrinkage 0.1; rebuilt lazily."""
        if self._W2 is not None:
            return self._W2
        d = len(self.mu2)
        C = self.cov_S2 / max(1, self.mu2_n) - np.outer(self.mu2,
                                                        self.mu2)
        C = 0.9 * C + 0.1 * np.eye(d, dtype=np.float32) \
            * max(C.trace(), 1e-6) / d
        w, V = np.linalg.eigh(C)
        self._W2 = ((V * (1.0 / np.sqrt(np.maximum(w, 1e-8)))) @ V.T
                    ).astype(np.float32)
        return self._W2

    def sem2_key(self, h):
        """The paper-8 semantic key: an early-layer hidden, centred on
        the v2 mean, ZCA-whitened when enabled, then banded. Pure --
        query-safe; statistics only move through sem2_observe."""
        z = h / (np.linalg.norm(h) + 1e-8)
        if self.mu2 is not None:
            z = z - self.mu2
        if self.sem2_whiten and self.mu2_n >= 256:
            z = z @ self._W2mat()
        z = z / (np.linalg.norm(z) + 1e-8)
        return self._bands(z)

    def sem2_pooled(self, H2, ids):
        """Paper 8's query: the tier's scores pooled over a whole prompt.

        No anchor heuristic survives a bare prompt -- every token of it
        looks surprising -- so the query is the position-wise maximum of
        the tier's scores over every prompt position, with the prompt's
        own tokens suppressed: recall never pays for what the window
        already holds.

        Both decoders call this, plain and speculative, which is what
        keeps `complete --fast` identical on a `--sem2` state.
        """
        pooled = None
        for k in range(0, len(H2), 64):
            Q = np.stack([self.sem2_key(H2[p_])
                          for p_ in range(k, min(k + 64, len(H2)))])
            U = Q @ self.MS
            S = (U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)) \
                @ self.V.T
            mx_ = S.max(axis=0)
            pooled = mx_ if pooled is None else np.maximum(pooled, mx_)
        if pooled is not None:
            pooled[list(set(int(t) for t in ids))] = -1e9
        return pooled

    @staticmethod
    def zca_of(X, shrink=0.1):
        """Whitening transform of a sample of hidden states."""
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        mu = Xn.mean(axis=0)
        C = np.cov((Xn - mu).T).astype(np.float32)
        d = C.shape[0]
        C = (1 - shrink) * C + shrink * np.eye(d, dtype=np.float32) \
            * max(float(C.trace()), 1e-6) / d
        w, V = np.linalg.eigh(C)
        W = ((V * (1.0 / np.sqrt(np.maximum(w, 1e-8)))) @ V.T
             ).astype(np.float32)
        return mu, W

    @staticmethod
    def sem2_separation(Aq, Ad, mu=None):
        """How well one layer keeps identity across contexts.

        `Aq` are hidden states of some tokens seen in SHORT queries,
        `Ad` the states of the same tokens seen inside the document.
        Same-token cosine minus cross-token cosine, centred and
        normalised. This is paper 8's measurement, and it is the one
        that matters: the tier is written from a document and queried
        from a prompt, so a layer that only looks stable WITHIN a
        document (the last one does, by predicting the same next word)
        is worthless here.
        """
        def prep(X):
            Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            if mu is not None:
                Z = Z - mu
            return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

        Q, D = prep(np.asarray(Aq)), prep(np.asarray(Ad))
        C = Q @ D.T
        same = np.diag(C)
        m = len(same)
        if m < 4:
            return None
        null = C[~np.eye(m, dtype=bool)]
        return float(np.median(same) - np.median(null))

    @staticmethod
    def sem2_score_layers(hiddens, ids, rng=None):
        """Which layer still knows WHO is being talked about?

        Paper 8 measured this with annotated entities; an installed tool
        has none, but it does not need any: a RARE TOKEN REPEATED in a
        document is the same identity in two different contexts, and two
        different rare tokens are the null. The separation between those
        two cosine distributions is what collapses with depth, so
        maximising it over layers finds the layer to key on, and its size
        says whether the geometry needs whitening.

        `hiddens` is a list of (n, d) arrays, one per layer. Returns the
        per-layer separations (None if the sample has too few repeats).
        """
        from collections import Counter
        ids = np.asarray(ids)
        counts = Counter(int(t) for t in ids)
        reps = [t for t, k in counts.items() if 2 <= k <= 20]
        if len(reps) < 8:
            return None
        rng = np.random.default_rng(0) if rng is None else rng
        same_pairs, pos_of = [], {}
        n_pos = len(ids)

        def free_of_context(p1, p2):
            """Two occurrences measure identity only if their
            NEIGHBOURS
            differ. Otherwise the last layer -- which encodes what comes
            next -- scores them identical for having the same
            continuation, and the sweep would crown the very layer
            paper 8 showed to be empty."""
            for d in (-1, 1):
                a, b = p1 + d, p2 + d
                if 0 <= a < n_pos and 0 <= b < n_pos and ids[a] == ids[b]:
                    return False
            return True

        for t in reps:
            p = np.flatnonzero(ids == t)
            pos_of[t] = p
            kept = 0
            for a in range(len(p) - 1):
                for b in range(a + 1, len(p)):
                    if kept >= 3:
                        break
                    if free_of_context(int(p[a]), int(p[b])):
                        same_pairs.append((int(p[a]), int(p[b])))
                        kept += 1
                if kept >= 3:
                    break
        flat = [(t, int(p)) for t in reps for p in pos_of[t]]
        null_pairs = []
        for _ in range(len(same_pairs)):
            (t1, p1), (t2, p2) = (flat[rng.integers(len(flat))],
                                  flat[rng.integers(len(flat))])
            if t1 != t2:
                null_pairs.append((p1, p2))
        if not same_pairs or not null_pairs:
            return None
        out = []
        for H in hiddens:
            Z = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
            Z = Z - Z.mean(axis=0, keepdims=True)
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            s = np.median([float(Z[a] @ Z[b]) for a, b in same_pairs])
            nl = np.median([float(Z[a] @ Z[b]) for a, b in null_pairs])
            out.append(float(s - nl))
        return out

    def sem2_flush(self, anchors, pairs, nulls):
        """Consolidate the v2 tier at the end of a document: apply the
        buffered writes with mature keys, then sample the null those
        queries are thresholded against. Both read paths call this, so
        a fast read and a normal one build the same tier."""
        for ai, tok_next, gw in pairs:
            q = self.sem2_key(anchors[ai])
            u = self.MS.T @ q          # scores() would also build a
            self.amp_write(self.MS, q, u,   # vocabulary vector we never
                           tok_next, max(gw, 0.25))          # read
        for k in range(0, len(nulls), 64):      # batched: one BLAS call
            blk = nulls[k:k + 64]               # per 64 nulls
            Q = np.stack([self.sem2_key(h) for h in blk])
            U = Q @ self.MS
            S = (U / (np.linalg.norm(U, axis=1, keepdims=True)
                      + 1e-8)) @ self.V.T
            self.res_S.extend(S.max(axis=1).tolist())

    def phi(self, h):
        """Fixed random feature of the hidden state, for the adapter."""
        v = h @ self.Rf(len(h))
        return v / (np.linalg.norm(v) + 1e-8)

    # ------------------------------------------------------- read ------------
    @staticmethod
    def _thr(res, q):
        """Confidence threshold: abstain below this quantile of the stream.

        q is None when calibration decided the tier should always speak.
        Below 500 observations the distribution is not known yet, so the
        tier stays silent rather than guessing.
        """
        if q is None:
            return -np.inf
        if len(res) < 500:
            return np.inf
        return float(np.quantile(res, q))

    def thresholds(self):
        """Current abstention thresholds for the two fast tiers."""
        return (self._thr(self.res_G, self.thr_qG),
                self._thr(self.res_S, self.thr_qS))

    def scores(self, M, q):
        """Read a tier: the retrieved vector, and a score for every token."""
        u = M.T @ q
        un = float(np.linalg.norm(u)) + 1e-8
        return u, (self.V @ u) / un

    # ------------------------------------------------- readout calibration --
    def collecting(self):
        """Should this position join the calibration window?

        One position in three, always -- the window ROLLS. A readout fitted
        once and frozen for good would be wrong twice over: what suits an
        empty matrix is far too timid for a full one, and the first document
        a fresh memory reads is exactly the coldest it will ever be. Keeping
        the most recent CALIB_MAX observations and refitting at the end of
        every read costs one log-sum-exp grid per three tokens and keeps the
        readout in step with what the memory actually knows.
        """
        return (self.calibrate_on
                and (self.tokens % CALIB_EVERY == 0))

    def collect(self, p_base_true, truth, sG, sS=None):
        """One dev observation: what each tier scored, before it was used.

        Costs one log-sum-exp per beta in the grid, at one position in
        three, for as long as calibration stays on: the window ROLLS (see
        `collecting`), so a state that keeps calibrating keeps paying this
        small tax in exchange for a readout that tracks the memory as it
        warms. If the semantic tier is switched on or off mid-window, the
        window restarts: half-filled semantic lists would otherwise
        misalign with "p" as the window rolls, and a readout must anyway be
        fitted on observations made under the mode it will govern.
        """
        sem = sS is not None
        if self.cal is not None and self.cal.get("sem") != sem:
            self.cal = None            # tier mode changed: fresh window
        if self.cal is None:
            self.cal = {"sem": sem, "p": [], "gt": [], "gm": [], "gl": [],
                        "st": [], "sm": [], "sl": []}
        c = self.cal
        c["p"].append(float(p_base_true))
        c["gt"].append(float(sG[truth]))
        c["gm"].append(float(sG.max()))
        c["gl"].append(lse_grid(sG))
        if sem:
            c["st"].append(float(sS[truth]))
            c["sm"].append(float(sS.max()))
            c["sl"].append(lse_grid(sS))
        if len(c["p"]) > CALIB_MAX:          # keep the most recent window
            for key in ("p", "gt", "gm", "gl", "st", "sm", "sl"):
                if c[key]:
                    del c[key][0]

    def maybe_calibrate(self):
        """Fit (beta, lambda, threshold) per tier on the window just closed.

        This is the tuning protocol of the papers, run inside the tool: the
        published grids, searched on held-out observations, then frozen for
        everything that follows. What is held out here is the past: the fit
        happens at the end of a read, on positions that were all scored with
        the PREVIOUS settings, and the winner governs the next read. So no
        perplexity this tool prints was ever tuned on itself. Tiers are
        fitted in the order they speak, each on top of what the previous one
        already produced.
        """
        if not self.calibrate_on:
            return None
        c = self.cal
        if not c or len(c["p"]) < CALIB_MIN:
            return None
        refit = self.calibrated
        p = np.array(c["p"], dtype=np.float64)
        report = {"n": len(p), "nll_before": float(
            -np.log(np.maximum(p, 1e-30)).mean())}
        nll, beta, lam, q, p_after = fit_readout(
            p, np.array(c["gt"], np.float32), np.array(c["gm"], np.float32),
            np.array(c["gl"], np.float32))
        self.beta_G, self.lam_G, self.thr_qG = beta, lam, q
        report["ngram"] = {"beta": beta, "lam": lam, "thr_q": q, "nll": nll}
        if self.semantic and len(c["st"]) == len(c["p"]):
            nll, beta, lam, q, p_after = fit_readout(
                p_after, np.array(c["st"], np.float32),
                np.array(c["sm"], np.float32), np.array(c["sl"], np.float32))
            self.beta_S, self.lam_S, self.thr_qS = beta, lam, q
            report["semantic"] = {"beta": beta, "lam": lam, "thr_q": q,
                                  "nll": nll}
        report["nll_after"] = float(report.get("semantic", report["ngram"])
                                    ["nll"])
        report["refit"] = refit
        self.calibrated = True
        self.cal_at = self.tokens
        return report                    # the window stays, and keeps rolling

    def recalibrate(self):
        """Drop the fitted readout and the window, and start again."""
        self.calibrated = False
        self.cal_at = 0
        self.cal = None
        self.calibrate_on = True
        self.beta_G, self.lam_G = DEFAULT_G
        self.beta_S, self.lam_S = DEFAULT_S
        self.thr_qG = self.thr_qS = DEFAULT_THR_Q

    def cold_lookup(self, tok_next=None):
        """Exact continuation statistics for the current n-gram, if any."""
        if len(self._hist) < NGRAM:
            return None
        gram = np.array(self._hist[-NGRAM:], dtype=np.int32).tobytes()
        slot = self.cold.get(gram)
        if slot is None or sum(slot[1].values()) < COLD_MIN_COUNT:
            return None
        # serve the distribution from per-successor surprise MASS when the
        # paper-6 fix is enabled (admission stays on raw counts either way,
        # so the two-occurrence rule is untouched); counts are the
        # historical behaviour and reproduce the papers' numbers
        src = (slot[2] if self.cold_mass and len(slot) > 2 and slot[2]
               else slot[1])
        tot = float(sum(src.values()))
        if tok_next is not None:
            return src.get(int(tok_next), 0) / tot
        return {t: c / tot for t, c in src.items()}

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
        """Mixed full distribution (generation time).

        Records in ``self.last_src`` the names of the tiers that actually
        cleared their threshold. A caller that also compares the argmax
        before and after can then tell a token the memory *produced*
        from one the frozen model invented on its own -- which is the
        difference between a recall and a fabrication that merely looks
        sourced.
        """
        p = p_base
        src = []
        for name, s, beta, lam, thr in (
                ("ngram", sG, self.beta_G, self.lam_G, thrG),
                ("sem", sS, self.beta_S, self.lam_S, thrS)):
            if s is None:
                continue
            if float(s.max()) >= (thr if thr is not None else np.inf):
                src.append(name)
                m = beta * s
                pm = np.exp(m - m.max())
                pm /= pm.sum()
                p = lam * pm + (1 - lam) * p
        if pc is not None:
            src.append("cold")
            p = (1 - LAM_C) * p
            for t, pv in pc.items():
                p[t] += LAM_C * pv
        self.last_src = src
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
            slot = self.cold.setdefault(gram, [0.0, {}, {}])
            if len(slot) == 2:         # slot written by a pre-1.2 state:
                slot.append({t: float(c) for t, c in slot[1].items()})
            slot[0] += g               # surprise mass, for consolidation
            slot[1][int(tok_next)] = slot[1].get(int(tok_next), 0) + 1
            slot[2][int(tok_next)] = slot[2].get(int(tok_next), 0.0) + g
        self.g_sum += g
        self.g_cnt += 1
        if p_adapted is not None:
            self.fw_update(phi, p_adapted, tok_next)
        self.tokens += 1
        self.decay_step()

    # ------------------------------------------------------- reporting -------
    def writes_per_parameter(self):
        """Paper 1's saturation coordinate: a fixed matrix fills near 0.5."""
        return self.tokens / float(D_K * D_V)

    def sizes(self):
        """Bytes held by each tier, in the order the papers introduce them."""
        return {
            "n-gram memory M_G": self.M.nbytes,
            "semantic memory M_S": self.MS.nbytes if self.semantic else 0,
            "cold store": sum(64 + 24 * len(v[1]) for v in self.cold.values()),
            "fast weights A": self.A.nbytes if self.fastweights else 0,
        }
