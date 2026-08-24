"""The assistant: a frozen language model plus the Sillage state.

    from sillage import Sillage
    s = Sillage(model="gpt2", state=".sillage")
    s.read("notes.md")                    # read + memorize, then save
    s.ask("what did the report say?")     # grounded excerpts
    print(s.complete("The protocol requires"))

Reading is strictly prequential: every token is scored with the memory as it
stands BEFORE that token is written, so the perplexity reported while reading
is an honest online measurement, never a replay of what was just stored.
Generation never writes: the assistant learns from what you give it to read,
not from its own output.
"""

import os
import time

import numpy as np

from .core import CAP, SillageMemory
from .index import Index, read_text

WINDOW, STRIDE = 1024, 512
DEFAULT_STATE = os.environ.get("SILLAGE_STATE", ".sillage")
PROGRESS_EVERY = 2000


def default_state():
    """`.sillage/` in the working directory, unless an older state is there."""
    if os.environ.get("SILLAGE_STATE"):
        return os.environ["SILLAGE_STATE"]
    if not os.path.exists(os.path.join(".sillage", "state.npz")) and \
            os.path.exists(os.path.join("memory_state", "state.npz")):
        return "memory_state"          # state written by pre-1.0 versions
    return ".sillage"


class Sillage:
    """A frozen model, its memory, and the index of what it has read.

    Everything the CLI does goes through this object, so the Python API and
    the command line cannot drift apart.
    """

    def __init__(self, model=None, state=None, semantic=None,
                 fastweights=None, half_life=None, calibrate=None,
                 quiet=False):
        self.state_dir = default_state() if state is None else state
        self.mem = SillageMemory(self.state_dir, model, semantic,
                                 fastweights, half_life, calibrate)
        self.index = Index(None if self.state_dir is None else
                           os.path.join(self.state_dir, "index.pkl"))
        self.quiet = quiet
        self._tok = None
        self._model = None

    # ------------------------------------------------------------- model ----
    def load_model(self):
        """Load the frozen model once, lazily: `ask` never needs it."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = self.mem.hub
            torch.set_num_threads(os.cpu_count() or 4)
            self._say(f"loading {name} (frozen) ...")
            try:
                self._tok = AutoTokenizer.from_pretrained(name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    name, dtype=torch.float32)
            except Exception as exc:
                raise SystemExit(
                    f"could not load {name} as a causal language model "
                    f"({type(exc).__name__}: {exc}).\nSillage needs a "
                    f"next-token predictor (GPT-2, Qwen, Llama, Mistral, "
                    f"Pythia, SmolLM ...), not an embedding or encoder "
                    f"model, and the weights must be reachable.")
            self._model.eval()
        return self._tok, self._model

    def _say(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    # -------------------------------------------------------------- read ----
    def read(self, *paths, save=True):
        """Read documents: memorize them and index them for grounded quotes."""
        stats = []
        for path in paths:
            name = os.path.basename(path)
            if any(f["file"] == name for f in self.mem.log["files"]):
                self._say(f"note: {name} was read before -- re-reading "
                          f"strengthens its traces.")
            text = read_text(path)
            n_pass = self.index.add(text, name)
            stats.append(self.read_text(text, name))
            stats[-1]["passages"] = n_pass
        if save:
            self.save()
        return stats

    def read_text(self, text, name="<text>"):
        """Stream one text through the frozen model and every memory tier."""
        import torch
        tok, model = self.load_model()
        mem = self.mem
        ids = np.array(tok.encode(text), dtype=np.int64)
        n = len(ids) - 1
        if n < 1:
            return {"file": name, "tokens": 0}
        mem.new_stream()
        thrG, thrS = mem.thresholds()
        need_h = mem.semantic or mem.fastweights
        nll_b = nll_f = nll_m = 0.0
        cnt = 0
        x = torch.tensor(ids)
        a = 0
        t0 = time.time()
        with torch.no_grad():
            while a < n:
                w = min(WINDOW, len(ids) - a)
                out = model(x[a:a + w].unsqueeze(0),
                            output_hidden_states=need_h)
                logits = out.logits[0].float().numpy()
                mem.set_vocab(logits.shape[-1])
                hs = (out.hidden_states[-1][0].float().numpy() if need_h
                      else None)
                lo = 0 if a == 0 else WINDOW - STRIDE
                for i in range(lo, w):
                    j = a + i
                    if j >= n:
                        break
                    truth = int(ids[j + 1])
                    lb = logits[i]
                    mx = lb.max()
                    lpb = lb - (mx + np.log(np.exp(lb - mx).sum()))
                    lp = float(lpb[truth])
                    # --- paper 4: adapted readout (scored before updating) ---
                    la, phi = mem.adapt(lb, hs[i] if need_h else None)
                    if phi is None:
                        p_ad, lp_f = None, lp
                    else:
                        m2 = la.max()
                        p_ad = np.exp(la - m2)
                        p_ad /= p_ad.sum()
                        lp_f = float(np.log(max(p_ad[truth], 1e-30)))
                    # --- papers 1-3: memory tiers, mixed at score level ------
                    qG = mem.step_key(int(ids[j]))
                    uG, sG = mem.scores(mem.M, qG)
                    mem.res_G.append(float(sG.max()))
                    qS = uS = sS = None
                    if mem.semantic:
                        qS = mem.sem_key(hs[i])
                        uS, sS = mem.scores(mem.MS, qS)
                        mem.res_S.append(float(sS.max()))
                    if mem.collecting():
                        # dev window: record what each tier would have said,
                        # to fit the readout to THIS model and THESE documents
                        mem.collect(np.exp(lp_f), truth, sG, sS)
                    pc = mem.cold_lookup(truth)
                    p = mem.mix_true(np.exp(lp_f), sG, truth, sS, pc,
                                     thrG, thrS)
                    nll_b += -lp
                    nll_f += -lp_f
                    nll_m += -np.log(max(p, 1e-30))
                    cnt += 1
                    # --- write: gate is the FROZEN model's own surprise ------
                    g = min(CAP, max(0.0, -lp))
                    mem.write_all(qG, uG, qS, uS, truth, g, phi, p_ad)
                    if cnt % PROGRESS_EVERY == 0:
                        rate = cnt / max(1e-6, time.time() - t0)
                        self._say(f"  ... {cnt}/{n} tokens "
                                  f"({(n - cnt) / rate / 60:.1f} min left)")
                if a + w >= len(ids):
                    break
                a += STRIDE
        mem.res_G = mem.res_G[-5000:]
        mem.res_S = mem.res_S[-5000:]
        calibration = mem.maybe_calibrate()
        rec = {"file": name, "tokens": int(cnt), "calibration": calibration,
               "date": time.strftime("%Y-%m-%d %H:%M"),
               "minutes": round((time.time() - t0) / 60, 1),
               "ppl_frozen": round(float(np.exp(nll_b / cnt)), 2),
               "ppl_fastweights": round(float(np.exp(nll_f / cnt)), 2),
               "ppl_with_memory": round(float(np.exp(nll_m / cnt)), 2)}
        mem.log["files"].append({k: rec[k] for k in
                                 ("file", "tokens", "date", "ppl_frozen",
                                  "ppl_fastweights", "ppl_with_memory")})
        return rec

    # ---------------------------------------------------------- generate ----
    def complete(self, prompt, n=40, temp=0.0, seed=0):
        """Continue a prompt with memory and fast weights. Writes nothing."""
        import torch
        tok, model = self.load_model()
        mem = self.mem
        ids = tok.encode(prompt)
        mem.new_stream()
        for t in ids[:-1]:
            mem.step_key(int(t))
        thrG, thrS = mem.thresholds()
        need_h = mem.semantic or mem.fastweights
        rng = np.random.default_rng(seed)
        past = None
        inp = torch.tensor(ids).unsqueeze(0)
        out_ids = []
        with torch.no_grad():
            for _ in range(n):
                out = model(inp, past_key_values=past, use_cache=True,
                            output_hidden_states=need_h)
                past = out.past_key_values
                lb = out.logits[0, -1].float().numpy()
                mem.set_vocab(lb.shape[-1])
                h = (out.hidden_states[-1][0, -1].float().numpy()
                     if need_h else None)
                la, _ = mem.adapt(lb, h)
                p_base = np.exp(la - la.max())
                p_base /= p_base.sum()
                qG = mem.step_key(int(inp[0, -1]))
                _, sG = mem.scores(mem.M, qG)
                sS = None
                if mem.semantic:
                    _, sS = mem.scores(mem.MS, mem.sem_key(h))
                p = mem.mix_full(p_base, sG, sS, mem.cold_lookup(),
                                 thrG, thrS)
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
        return tok.decode(out_ids)

    # --------------------------------------------------------------- ask ----
    def ask(self, question, k=3, numeric_only=False):
        """Grounded passages from what has been read. Nothing is generated."""
        return self.index.search(question, k=k, numeric_only=numeric_only)

    def add_to_index(self, path):
        """Index a document without reading it into the memory (instant)."""
        n = self.index.add(read_text(path), os.path.basename(path))
        self.index.save()
        return n

    # ------------------------------------------------------------ report ----
    def status(self):
        """Everything `sillage status` prints, as a dictionary."""
        mem = self.mem
        disk = 0
        for f in ("state.npz", "cold.pkl", "index.pkl", "log.json"):
            p = os.path.join(self.state_dir, f)
            if os.path.exists(p):
                disk += os.path.getsize(p)
        return {"model": mem.hub, "state_dir": self.state_dir,
                "tokens": mem.tokens,
                "documents": len({f["file"] for f in mem.log["files"]}),
                "cold_grams": len(mem.cold), "passages": len(
                    self.index.passages),
                "semantic": mem.semantic, "fastweights": mem.fastweights,
                "half_life": mem.half_life,
                "calibrated": mem.calibrated,
                "calibrating": mem.calibrate_on,
                "readout": {"ngram": (mem.beta_G, mem.lam_G, mem.thr_qG),
                            "semantic": (mem.beta_S, mem.lam_S, mem.thr_qS)},
                "calib_seen": 0 if not mem.cal else len(mem.cal["p"]),
                "writes_per_parameter": mem.writes_per_parameter(),
                "sizes": mem.sizes(), "disk": disk,
                "files": mem.log["files"]}

    def save(self):
        """Consolidate the memory and write both state and index to disk."""
        self.mem.save()
        self.index.save()
