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
                 device=None, quiet=False, target=None, cold_mass=None,
                 sem2=None, sem2_whiten=None):
        self.state_dir = default_state() if state is None else state
        self.target_hub = target
        if target is not None:
            # paper 5: a state serves any same-tokenizer sibling, but the
            # adapter is a function of the READING model's hidden geometry
            fastweights = False
        self.mem = SillageMemory(self.state_dir, model, semantic,
                                 fastweights, half_life, calibrate,
                                 cold_mass, sem2, sem2_whiten)
        self.index = Index(None if self.state_dir is None else
                           os.path.join(self.state_dir, "index.pkl"))
        self.quiet = quiet
        self.device = device        # None -> cuda when there is one
        self._tok = None
        self._model = None

    # ------------------------------------------------------------- model ----
    def load_model(self):
        """Load the frozen model once, lazily: `ask` never needs it."""
        if self._model is not None:
            if self.device is None:      # a model handed to us from outside
                self.device = str(next(self._model.parameters()).device)
            return self._tok, self._model
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = self.target_hub or self.mem.hub
            if self.target_hub:
                self._say(f"target {name} reading a state written by "
                          f"{self.mem.hub} (shared tokenizer required; "
                          f"adapter off)")
            if self.device is None:
                self.device = ("cuda" if torch.cuda.is_available()
                               else "cpu")
            torch.set_num_threads(os.cpu_count() or 4)
            self._say(f"loading {name} (frozen, {self.device}) ...")
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
            # the mechanisms are numpy on the CPU either way; the device only
            # decides where the frozen forward passes happen
            self._model.to(self.device)
            self._model.eval()
        return self._tok, self._model

    def _say(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    # -------------------------------------------------------------- read ----
    def read(self, *paths, save=True, fast=False):
        """Read documents: memorize them and index them for grounded quotes.

        fast=True is paper 7's blocked ingestion: writes only, ~40x on
        long documents, no perplexity report; the cold store is exact,
        amplitude tolerances are declared, and the adapter does not
        learn during a fast read (it still serves at generation).
        """
        if fast and self.mem.fastweights:
            self._say("fast read: the adapter does not learn during "
                      "this read (its delta rule is sequential); it "
                      "still serves at generation time.")
        stats = []
        for path in paths:
            name = os.path.basename(path)
            if any(f["file"] == name for f in self.mem.log["files"]):
                self._say(f"note: {name} was read before -- re-reading "
                          f"strengthens its traces.")
            text = read_text(path)
            n_pass = self.index.add(text, name)
            if fast:
                from .ingest import ingest_text
                stats.append(ingest_text(self, text, name,
                                         quiet=self.quiet))
            else:
                stats.append(self.read_text(text, name))
            stats[-1]["passages"] = n_pass
        if save:
            self.save()
        return stats

    def read_text(self, text, name="<text>"):
        """Stream one text through the frozen model and every memory tier.

        With the paper-8 semantic keys on (`sem2`), that tier writes at
        the END of the document rather than token by token, so the
        perplexity reported for this read does not include it.
        """
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
        sem2 = mem.sem2_layer if mem.semantic else None   # paper 8
        prev_kept, g_prev, anchor_idx = False, 0.0, None
        nsp = {}                             # token id -> continuation piece?
        # The tier's writes are BUFFERED and applied in bulk. The key
        # statistics (the mean, and the whitening when it is on) mature
        # as the document is read, and a key written against half-formed
        # statistics does not match the one the query will derive later.
        # Buffering also pays the whitening's eigendecomposition once
        # per flush instead of once per token.
        anchors, sem2_buf, null_buf = [], [], []
        null_stride = 4

        def flush_sem2():
            """Write the buffered pairs with the current (mature) keys,
            then sample the null those queries are thresholded against."""
            nonlocal anchor_idx
            for ai, tok_next, gw in sem2_buf:
                qs = mem.sem2_key(anchors[ai])
                us = mem.MS.T @ qs        # scores() would also build a
                mem.amp_write(mem.MS, qs, us, tok_next,   # vocabulary
                              max(gw, 0.25))              # vector we
                                                          # never read
            for k in range(0, len(null_buf), 64):     # batched: one
                blk = null_buf[k:k + 64]              # BLAS call per 64
                Q = np.stack([mem.sem2_key(h) for h in blk])
                U = Q @ mem.MS
                S = (U / (np.linalg.norm(U, axis=1, keepdims=True)
                          + 1e-8)) @ mem.V.T
                mem.res_S.extend(S.max(axis=1).tolist())
            last = None if anchor_idx is None else anchors[anchor_idx]
            anchors.clear()
            sem2_buf.clear()
            null_buf.clear()
            if last is not None:
                anchors.append(last)
                anchor_idx = 0

        nll_b = nll_f = nll_m = 0.0
        cnt = 0
        x = torch.tensor(ids, device=self.device)
        a = 0
        t0 = time.time()
        with torch.no_grad():
            while a < n:
                w = min(WINDOW, len(ids) - a)
                out = model(x[a:a + w].unsqueeze(0),
                            output_hidden_states=need_h)
                logits = out.logits[0].float().cpu().numpy()
                mem.set_vocab(logits.shape[-1])
                hs = (out.hidden_states[-1][0].float().cpu().numpy()
                      if need_h else None)
                hs2 = (out.hidden_states[sem2][0].float().cpu().numpy()
                       if sem2 is not None else None)
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
                    if sem2 is not None:
                        # paper 8: keys from the early layer, anchored
                        # on the last surprising token, writes filtered
                        # for surprise with word-integrity readmission
                        mem.sem2_observe(hs2[i])
                        if g_prev >= 2.5:      # this token surprised it
                            anchors.append(hs2[i].copy())
                            anchor_idx = len(anchors) - 1
                            if len(anchors) >= 8192:
                                flush_sem2()   # bound the buffer
                        g2 = min(CAP, max(0.0, -lp))
                        if truth not in nsp:
                            d2 = tok.decode([truth])
                            nsp[truth] = (len(d2) > 0
                                          and not d2[0].isspace())
                        keep = g2 >= 0.5 or (prev_kept and nsp[truth])
                        if anchor_idx is not None and keep:
                            sem2_buf.append((anchor_idx, truth, g2))
                        elif cnt % null_stride == 0:
                            null_buf.append(hs2[i].copy())
                            if len(null_buf) > 4096:
                                del null_buf[::2]
                                null_stride *= 2
                        prev_kept = keep
                    elif mem.semantic:
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
                    g_prev = g
                    if cnt % PROGRESS_EVERY == 0:
                        rate = cnt / max(1e-6, time.time() - t0)
                        self._say(f"  ... {cnt}/{n} tokens "
                                  f"({(n - cnt) / rate / 60:.1f} min left)")
                if a + w >= len(ids):
                    break
                a += STRIDE
        if sem2 is not None:
            flush_sem2()
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
    def complete(self, prompt, n=40, temp=0.0, seed=0, fast=False):
        """Continue a prompt with memory and fast weights. Writes nothing.

        fast=True verifies drafts from the memory in blocks (paper 5):
        greedy only, output identical to fast=False by construction --
        faster exactly where the memory is confident.
        """
        if fast:
            if temp and temp > 0:
                self._say("--fast is greedy-only (speculative sampling not "
                          "implemented); falling back to plain decoding.")
            else:
                from .drafting import complete_fast
                text, stats = complete_fast(self, prompt, n=n)
                acc = stats["accepted"] / max(1, stats["drafted"])
                self._say(f"  [fast: {stats['tokens']} tokens in "
                          f"{stats['forwards']} forwards, "
                          f"{stats['accepted']}/{stats['drafted']} drafts "
                          f"accepted ({acc:.0%})]")
                return text
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
        pooled = None            # paper 8: one pooled query per prompt
        inp = torch.tensor(ids, device=self.device).unsqueeze(0)
        out_ids = []
        with torch.no_grad():
            for step in range(n):
                out = model(inp, past_key_values=past, use_cache=True,
                            output_hidden_states=need_h)
                past = out.past_key_values
                lb = out.logits[0, -1].float().cpu().numpy()
                mem.set_vocab(lb.shape[-1])
                h = (out.hidden_states[-1][0, -1].float().cpu().numpy()
                     if need_h else None)
                la, _ = mem.adapt(lb, h)
                p_base = np.exp(la - la.max())
                p_base /= p_base.sum()
                qG = mem.step_key(int(inp[0, -1]))
                _, sG = mem.scores(mem.M, qG)
                sS = None
                if mem.sem2_layer is not None and mem.semantic:
                    if pooled is None:
                        # paper 8: pool the tier over every prompt
                        # position (no anchor heuristic survives a bare
                        # prompt), then suppress the echo -- recall
                        # never pays for what the window already holds
                        H2 = (out.hidden_states[mem.sem2_layer][0]
                              .float().cpu().numpy())
                        for k in range(0, len(H2), 64):
                            Q = np.stack([mem.sem2_key(H2[p_])
                                          for p_ in
                                          range(k, min(k + 64,
                                                       len(H2)))])
                            U = Q @ mem.MS
                            S = (U / (np.linalg.norm(U, axis=1,
                                                     keepdims=True)
                                      + 1e-8)) @ mem.V.T
                            mx_ = S.max(axis=0)
                            pooled = (mx_ if pooled is None
                                      else np.maximum(pooled, mx_))
                        pooled[list(set(int(t) for t in ids))] = -1e9
                    # the v2 tier gives ONE impulse: it recalls the
                    # value's head, and the frozen model finishes the
                    # word from there. Sustained mixing recalls no more
                    # (measured) and disturbs unrelated prompts twice as
                    # often. The n-gram tier continues; this one recalls.
                    sS = pooled if step == 0 else None
                elif mem.semantic:
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
                inp = torch.tensor([[nxt]], device=self.device)
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
                "sem2_layer": mem.sem2_layer,
                "sem2_whiten": mem.sem2_whiten,
                "half_life": mem.half_life, "cold_mass": mem.cold_mass,
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
