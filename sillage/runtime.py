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

import json
import os
import time

import numpy as np

from .core import CAP, SEM2_LAYER, SEM2_WHITEN, SillageMemory
from .index import Index, read_text

WINDOW, STRIDE = 1024, 512
PROGRESS_EVERY = 2000
# how often an ingestion offers its lock back to a server: a
# reader gives up its turn every this many tokens, so a chat
# request waits for a few tokens of writing, not a whole window
YIELD_EVERY = 32


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
                 sem2=None, sem2_whiten=None, dtype=None):
        self.state_dir = (default_state() if state is None
                          else os.path.expanduser(state))
        self.target_hub = target
        if target is not None:
            # paper 5: a state serves any same-tokenizer sibling, but the
            # adapter is a function of the READING model's hidden geometry
            fastweights = False
        # kept so `pull` can rebuild the memory for the model a cartridge
        # declares, exactly as if it had been opened that way
        self._mem_arg = (model, (semantic, fastweights, half_life,
                                 calibrate, cold_mass, sem2, sem2_whiten))
        self.mem = SillageMemory(self.state_dir, model, semantic,
                                 fastweights, half_life, calibrate,
                                 cold_mass, sem2, sem2_whiten)
        self.index = Index(None if self.state_dir is None else
                           os.path.join(self.state_dir, "index.json"))
        self.quiet = quiet
        self.device = device        # None -> cuda when there is one
        self.dtype = dtype          # None -> float32, or int8/bf16/fp16
        self._tok = None
        self._model = None

    # ------------------------------------------------------------- model ----
    def _check_hidden_width(self):
        """Silence the tiers a different-width reader cannot key.

        Paper 5's transfer -- read with the small model, serve with a
        bigger sibling -- holds for the tiers keyed on TOKENS: the
        n-gram matrix and the cold store are indexed by the last four
        token ids, and the family shares a tokenizer. It does NOT hold
        for the tiers keyed on HIDDEN STATES: the v1 semantic centre
        `mu` and paper 8's `mu2` have the width of the model that wrote
        them (1024 for Qwen3-0.6B), and a 1.7B hands them 2048.

        Until 1.9.1 that mismatch raised `operands could not be
        broadcast together with shapes (2048,) (1024,)` from inside a
        decoding loop -- so `complete --target`, the headline of paper
        5, crashed on any state with the semantic tier on, which is the
        default for qwen. Shipped since 1.1.0 and never caught, because
        every measurement of the transfer built its state WITH the
        target model, where the widths agree.

        The tiers that cannot transfer now abstain and say so; the ones
        that can keep working, which is what the paper actually claims.
        """
        mem = self.mem
        try:
            width = int(self._model.config.hidden_size)
        except Exception:
            return
        lost = []
        if mem.semantic and mem.mu is not None and len(mem.mu) != width:
            mem.semantic = False
            lost.append(f"paper 2's semantic tier (built at "
                        f"{len(mem.mu)}d)")
        if (mem.sem2_layer is not None and mem.mu2 is not None
                and len(mem.mu2) != width):
            mem.sem2_layer = None
            lost.append(f"paper 8's tier (built at {len(mem.mu2)}d)")
        if lost:
            self._say(
                f"this reader's hidden states are {width}d, so "
                f"{' and '.join(lost)} cannot be keyed and stay "
                f"silent.\n  The n-gram matrix and the cold store are "
                f"keyed on tokens and transfer normally (paper 5).")

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
            # every core by default, but not against the user's wishes:
            # a read runs for minutes, and `OMP_NUM_THREADS=4 sillage read`
            # is how you keep the rest of the machine usable.
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS")
                                      or os.cpu_count() or 4))
            self._say(f"loading {name} (frozen, {self.device}) ...")
            # A bigger model on modest hardware, without giving up the
            # hidden states the memory keys on (paper 8) -- which is why
            # this is torch's own quantisation and not a GGUF runtime.
            want = (self.dtype or "float32").lower()
            load_as = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16}.get(want,
                                                     torch.float32)
            try:
                self._tok = AutoTokenizer.from_pretrained(name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    name, dtype=load_as)
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
            self._check_hidden_width()
            if want == "int8":
                # dynamic int8 over the Linear layers: CPU-native, in
                # torch itself (no extra dependency), and the forward
                # still returns hidden states. GPT-2 builds its blocks
                # from transformers' Conv1D, which this leaves alone --
                # the tool says how little it touched instead of
                # reporting a layer count that flatters.
                n_lin = sum(1 for m in self._model.modules()
                            if isinstance(m, torch.nn.Linear))
                # GPT-2 and its descendants build their blocks from
                # transformers' own Conv1D, which dynamic quantisation
                # does not touch. Say how much was actually left alone
                # rather than reporting a layer count that flatters.
                n_conv = sum(1 for m in self._model.modules()
                             if type(m).__name__ == "Conv1D")
                if n_lin == 0:
                    self._say(
                        f"--dtype int8: this architecture has no "
                        f"nn.Linear layers to quantise ({n_conv} Conv1D "
                        f"instead, as GPT-2 does); loading it "
                        f"unquantised.")
                else:
                    self._model = torch.ao.quantization.quantize_dynamic(
                        self._model, {torch.nn.Linear},
                        dtype=torch.qint8)
                    if n_conv:
                        self._say(
                            f"--dtype int8: quantised {n_lin} nn.Linear "
                            f"layer(s), but {n_conv} of this model's "
                            f"layers are transformers' Conv1D, which "
                            f"dynamic quantisation leaves alone -- so "
                            f"almost nothing was saved here.")
                    self._say(
                        f"quantised {n_lin} linear layer(s) to int8 "
                        f"(dynamic, CPU). MEASURED on Qwen3-0.6B: the "
                        f"cold store's admissions come out identical "
                        f"(Jaccard 1.00) but the surprise gate only "
                        f"correlates 0.97 with float32, recall dropped "
                        f"from 5/7 to 1/7, and reading was no faster. "
                        f"Use this to fit a bigger model in memory, not "
                        f"to go faster or to read something you care "
                        f"about.")
            if want not in ("float32", "int8"):
                self._say(
                    f"loaded in {want}: half the weight memory, and "
                    f"MEASURED faithful on Qwen3-0.6B (the surprise "
                    f"gate correlates 1.000 with float32, identical "
                    f"cold admissions, identical recall) -- but on a "
                    f"CPU without native support it is emulated, and "
                    f"reading measured 4x slower. Worth it when memory "
                    f"is the constraint, not when time is.")
        return self._tok, self._model

    def load_tokenizer(self):
        """The tokenizer alone. `review` counts n-grams and `export` copies
        matrices: neither runs a forward pass, and loading half a gigabyte
        of weights to do it cost twelve seconds and a progress bar."""
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(
                self.target_hub or self.mem.hub)
        return self._tok

    def _say(self, msg):
        if not self.quiet:
            print(msg, flush=True)

    # -------------------------------------------------------------- read ----
    @staticmethod
    def reflow(text):
        """Join the lines inside each paragraph, keeping the breaks between.

        Both fast tiers key on the last four TOKENS, and a line break is
        absorbed INTO a token: a document that wraps `responsable,` /
        `madame` stores the key `[' responsable', ',\\n', 'mad', 'ame']`,
        while the same sentence typed on one line asks for
        `[' responsable', ',', ' mad', 'ame']`. The store misses, both
        tiers stay silent, and the frozen model fills the silence with a
        fabrication -- which is what `Brigitte Lefevre` was.

        Measured on a 430-token French report, questions typed the way a
        person types them: 7/8 facts as the document wraps them, **8/8**
        reflowed (and 7/8 reflowed with the semantic tier off -- the
        cold store opens the name, that tier finishes it).

        It is OPT-IN because it changes the token stream, so a reflowed
        read is not comparable to the published perplexities: on that
        report, frozen 22.79 -> 24.66 and with-memory 18.28 -> 19.69.
        Recall of questions and fidelity to the document's own shape are
        genuinely different goals; this picks the first, on request.
        """
        import re
        paras = re.split(r"\n\s*\n", text)
        return "\n\n".join(" ".join(p.split()) for p in paras if p.strip())

    def read(self, *paths, save=True, fast=False, between_windows=None,
             name=None, reflow=False):
        """Read documents: memorize them and index them for grounded quotes.

        fast=True is paper 7's blocked ingestion: writes only, ~40x on
        long documents, no perplexity report; the cold store is exact,
        amplitude tolerances are declared, and the adapter does not
        learn during a fast read (it still serves at generation).

        `name` overrides the key a document is known by, and is only
        valid for a single path. `sillage watch` passes the path
        relative to the folder it walks, so two `notes.md` in two
        subfolders do not evict each other from the index; `review
        --read` passes the key a document already has, so a reread
        updates it instead of creating a second entry.
        """
        if name is not None and len(paths) != 1:
            raise ValueError("name= applies to a single path")
        if fast and self.mem.fastweights:
            self._say("fast read: the adapter does not learn during "
                      "this read (its delta rule is sequential); it "
                      "still serves at generation time.")
        stats = []
        for path in paths:
            path = os.path.expanduser(path)
            key = name or os.path.basename(path)
            if any(f["file"] == key for f in self.mem.log["files"]):
                self._say(f"note: {key} was read before -- re-reading "
                          f"strengthens its traces.")
            text = read_text(path)
            if reflow:
                text = self.reflow(text)
            n_pass = self.index.add(text, key)
            if fast:
                from .ingest import ingest_text
                stats.append(ingest_text(
                    self, text, key, quiet=self.quiet,
                    between_windows=between_windows))
            else:
                stats.append(self.read_text(
                    text, key, between_windows=between_windows))
            stats[-1]["passages"] = n_pass
            # keep the path so `review --read` can reread a document that
            # does not happen to sit in the current working directory
            stats[-1]["path"] = os.path.abspath(path)
            if self.mem.log["files"]:
                self.mem.log["files"][-1]["path"] = os.path.abspath(path)
        if save:
            self.save()
        return stats

    @staticmethod
    def _centre(v, mu):
        z = v / (np.linalg.norm(v) + 1e-8) - mu
        return z / (np.linalg.norm(z) + 1e-8)

    def resolve_sem2(self, ids):
        """Pick the layer to key the v2 tier on, from the document being
        read (paper 8's sweep, run on free supervision: rare repeated
        tokens). Called once, then the choice lives in the state."""
        import torch
        mem = self.mem
        if not mem.sem2_auto or mem.sem2_layer is not None:
            return
        if mem.which in SEM2_LAYER:      # measured beats swept, as it
            mem.sem2_layer = SEM2_LAYER[mem.which]        # does for the
            if mem._sem2_whiten_arg is None:              # readout
                mem.sem2_whiten = SEM2_WHITEN[mem.which]
            mem.sem2_auto = False
            self._say(f"--sem2 auto: layer {mem.sem2_layer} and "
                      f"whitening {'on' if mem.sem2_whiten else 'off'} "
                      f"-- what paper 8 measured for {mem.which}.")
            return
        tok, model = self.load_model()
        sample = np.asarray(ids[:min(len(ids), WINDOW)])   # one window:
        #        a short-context model refuses anything longer
        with torch.no_grad():
            out = model(torch.tensor(sample, device=self.device
                                     ).unsqueeze(0),
                        output_hidden_states=True)
        H = [h[0].float().cpu().numpy() for h in out.hidden_states]
        # Free supervision, paper 8's protocol without annotation: a
        # rare token repeated in the document is one identity, and a
        # SHORT WINDOW ending on one of its occurrences is a query for
        # it. Comparing the two measures exactly what the tier needs --
        # invariance between a document and a prompt.
        from collections import Counter
        counts = Counter(int(t) for t in sample)
        reps = [t for t, k in counts.items() if 2 <= k <= 20][:32]
        pairs = []
        for t in reps:
            p = np.flatnonzero(sample == t)
            if len(p) >= 2 and p[-1] >= 12:
                pairs.append((int(p[0]), int(p[-1])))     # doc, query
        seps = None
        if len(pairs) >= 6:
            probes = np.stack([sample[q - 11:q + 1] for _, q in pairs])
            with torch.no_grad():
                po = model(torch.tensor(probes, device=self.device),
                           output_hidden_states=True)
            seps = []
            for li, h in enumerate(po.hidden_states):
                Aq = h[:, -1].float().cpu().numpy()
                Ad = np.stack([H[li][d] for d, _ in pairs])
                Hn = H[li] / (np.linalg.norm(H[li], axis=1,
                                             keepdims=True) + 1e-8)
                seps.append(mem.sem2_separation(Aq, Ad,
                                                Hn.mean(axis=0)))
            if any(v is None for v in seps):
                seps = None
        if seps is None:
            mem.sem2_auto = False
            self._say("--sem2 auto: this text repeats too few rare "
                      "tokens to choose a layer; the tier stays off. "
                      "Give it a longer document, or name the layer.")
            mem.semantic = False
            return
        # layer 0 is the embedding table: identity is trivially perfect
        # there and carries no context, so the sweep starts at 1
        best = int(np.argmax(seps[1:])) + 1
        mem.sem2_layer = best
        why = ""
        if mem._sem2_whiten_arg is None:
            # do not guess from the size of the separation -- measure:
            # whiten this layer's sample and see whether it separates
            # better. One eigendecomposition, once per state.
            # The LAYER can be found from the document; whether the
            # geometry needs whitening cannot -- every cheap proxy we
            # tried (cosine separation, throwaway-tier retrieval rank)
            # says "no" for GPT-2, which paper 8 measured as needing it
            # badly. Both proxies compare two places in one document,
            # and that is not the question. So we use what was measured
            # per model, and default to whitening for an unmeasured one
            # (paper 2's rule: raw hidden states need it except where
            # the geometry is already well conditioned).
            mem.sem2_whiten = SEM2_WHITEN.get(mem.which, True)
            why = (f"measured for {mem.which}"
                   if mem.which in SEM2_WHITEN
                   else "the safe default for a model nobody "
                        "has measured")
        mem.sem2_auto = False
        self._say(f"--sem2 auto: layer {best} keeps entity identity best "
                  f"here (separation {seps[best]:+.2f}, last layer "
                  f"{seps[-1]:+.2f}); whitening "
                  f"{'on' if mem.sem2_whiten else 'off'}"
                  + ("" if mem._sem2_whiten_arg is not None else
                     f" ({why}; --sem2-whiten / --no-sem2-whiten "
                     f"overrides)") + ".")

    def read_text(self, text, name="<text>", between_windows=None):
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
        self.resolve_sem2(ids)
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
            """Consolidate the tier (core.sem2_flush), then keep the
            live anchor so the stream continues across the flush."""
            nonlocal anchor_idx
            mem.sem2_flush(anchors, sem2_buf, null_buf)
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
                    if between_windows is not None and cnt % YIELD_EVERY == 0:
                        # a server yields its lock HERE too, not only at
                        # the window boundary: a document that fits in one
                        # 1024-token window has no boundary at all, so a
                        # conversation used to stall for the whole read.
                        # The state is consistent at every token boundary
                        # -- the write for this token is already done.
                        between_windows()
                if between_windows is not None:
                    between_windows()   # and once per window, as before
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
    def attribution(self):
        """What the memory contributed to the last ``complete``.

        Returns ``None`` when no trace was kept, else a dict with the
        number of generated tokens, how many the memory *moved* (the top
        choice differs with and without it) and which tiers spoke. A run
        with ``moved == 0`` is the frozen model talking alone: whatever
        it said, the memory did not say it.
        """
        tr = getattr(self, "last_trace", None)
        if tr is None:
            return None
        tiers = {}
        for src, _ in tr:
            for s in src:
                tiers[s] = tiers.get(s, 0) + 1
        return {"tokens": len(tr),
                "moved": sum(1 for _, m in tr if m),
                "spoke": sum(1 for s, _ in tr if s),
                "tiers": tiers}

    def complete(self, prompt, n=40, temp=0.0, seed=0, fast=False,
                 on_token=None):
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
                self.last_trace = None   # the block verifier keeps no trace
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
        trace = []
        self.last_trace = trace
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
                        pooled = mem.sem2_pooled(
                            out.hidden_states[mem.sem2_layer][0]
                            .float().cpu().numpy(), ids)
                    # the v2 tier gives ONE impulse: it recalls the
                    # value's head, and the frozen model finishes the
                    # word from there. Sustained mixing recalls no more
                    # (measured) and disturbs unrelated prompts twice as
                    # often. The n-gram tier continues; this one recalls.
                    sS = pooled if step == 0 else None
                elif mem.semantic:
                    # learn=False: answering is not reading (see sem_key)
                    _, sS = mem.scores(mem.MS, mem.sem_key(h, learn=False))
                p = mem.mix_full(p_base, sG, sS, mem.cold_lookup(),
                                 thrG, thrS)
                # attribution: which tiers spoke, and did they actually
                # move the model's top choice. The second is the honest
                # one -- a tier can clear its threshold and change
                # nothing, and a token the memory did not move is the
                # frozen model's invention, not a recall.
                trace.append((list(mem.last_src),
                              int(np.argmax(p)) != int(np.argmax(p_base))))
                if temp and temp > 0:
                    logp = np.log(np.maximum(p, 1e-30)) / temp
                    pp = np.exp(logp - logp.max())
                    pp /= pp.sum()
                    nxt = int(rng.choice(len(pp), p=pp))
                else:
                    nxt = int(np.argmax(p))
                out_ids.append(nxt)
                if on_token is not None:
                    # the whole text so far, decoded: a caller streaming
                    # this computes its own delta, because a subword token
                    # is not a printable increment on its own
                    on_token(tok.decode(out_ids))
                inp = torch.tensor([[nxt]], device=self.device)
                if nxt == getattr(tok, "eos_token_id", -1):
                    break
        return tok.decode(out_ids)

    # ---------------------------------------------------------- sharing ----
    def export_shareable(self, out_dir):
        """Write a state someone else can open without receiving your
        documents.

        What is left out, and why it has to be: the COLD STORE is a
        table of 4-grams to their successors, in plain token ids, and
        the INDEX keeps whole passages verbatim -- either one hands the
        reader your text back. What are kept are the matrices, which are
        superpositions: measured on the state paper 8 was written from,
        dropping both costs no perplexity (1.20 -> 1.19) and two
        canonical recalls in ten (9/10 -> 7/10), and leaves paraphrased
        recall untouched at 8/10 -- so a cartridge is worth sharing.

        This is *no plain text*, which is not the same claim as
        *anonymous*: inverting a superposition is hard, not proven
        impossible, and no attack has been run against it yet. The
        manifest says so, in the file.
        """
        os.makedirs(out_dir, exist_ok=True)
        # A cartridge speaks through the matrices alone, and a tier that
        # has not seen 500 scored positions abstains by design (core._thr
        # will not guess a quantile from less). Say so before writing a
        # silent file: measured, a state with ~1.9k tokens read fast
        # recalls nothing without its cold store, while one with ~7k
        # recalls 10 facts in 10 -- the same as the full state.
        thin = len(self.mem.res_G) < 500
        keep = self.mem.cold
        self.mem.cold = {}                 # not written, not pruned
        try:
            here = self.state_dir
            self.mem.dir = out_dir
            self.mem.save()
        finally:
            self.mem.dir = here
            self.mem.cold = keep
        for stray in ("cold.npz", "calib.npz", "index.json"):
            p = os.path.join(out_dir, stray)
            if os.path.exists(p):
                os.remove(p)
        # the log records the absolute path each document was read from --
        # useful locally for `review --read`, and nobody else's business.
        # Strip it: a cartridge that ships someone's home directory and
        # their filenames is not "no plain text".
        log_p = os.path.join(out_dir, "log.json")
        if os.path.exists(log_p):
            with open(log_p, encoding="utf-8") as f:
                log = json.load(f)
            for rec in log.get("files", []):
                rec.pop("path", None)
            with open(log_p, "w", encoding="utf-8") as f:
                json.dump(log, f)
        from . import __version__
        manifest = {
            "sillage": __version__,
            "model": self.mem.which, "hub": self.mem.hub,
            "vocab": int(self.mem.vocab),
            "tokens_read": int(self.mem.tokens),
            # documents, not read events: a file read twice is one document
            "documents": sorted({f["file"] for f in self.mem.log["files"]}),
            "reads": len(self.mem.log["files"]),
            "semantic": bool(self.mem.semantic),
            "sem2_layer": self.mem.sem2_layer,
            "sem2_whiten": bool(self.mem.sem2_whiten),
            # declared on, and separately whether it holds anything: a
            # state built by `watch` (fast reads) ships this tier empty
            "fastweights": bool(self.mem.fastweights),
            "fastweights_written": bool(self.mem.fastweights
                                        and float(np.abs(self.mem.A).max())
                                        > 0),
            "left_out": ["cold store (plain token n-grams)",
                         "index (verbatim passages)",
                         "readout calibration window",
                         "the paths the documents were read from"],
            "caveat": ("no plain text is included; that is not the same "
                       "as anonymous -- inverting the matrices is hard, "
                       "not proven impossible, and no inversion attack "
                       "has been run against this format yet"),
        }
        with open(os.path.join(out_dir, "cartridge.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        total = sum(os.path.getsize(os.path.join(out_dir, f))
                    for f in os.listdir(out_dir))
        return {"dir": out_dir, "bytes": total, "manifest": manifest,
                "files": sorted(os.listdir(out_dir)),
                "thin": thin, "scored": len(self.mem.res_G)}

    # ------------------------------------------------------------- pull ----
    CARTRIDGE_FILES = ("cartridge.json", "state.npz", "log.json")

    def pull_cartridge(self, source, force=False):
        """Open somebody else's cartridge as this state directory.

        The other half of `export_shareable`. Three rules, and each one
        is a refusal rather than a warning, because what arrives here
        came from a stranger:

        * only the three files a cartridge is made of are ever copied,
          by name -- never a whole repository, so nothing else rides in;
        * a pre-1.5 pickle is refused outright. Our own states migrate
          with a warning, but unpickling executes code, and a downloaded
          state is not one you created;
        * an existing memory is never silently overwritten.

        `source` is a local directory or a Hugging Face repo id
        (`user/name`, model repo first, then dataset).
        """
        import shutil
        dest = self.state_dir
        if os.path.isdir(dest) and not force:
            busy = [f for f in os.listdir(dest)
                    if f.endswith((".npz", ".json", ".pkl"))]
            if busy:
                raise RuntimeError(
                    "%s already holds a memory (%s). Pulling would "
                    "replace it: pass --force, or --state DIR to keep "
                    "both." % (dest, ", ".join(sorted(busy)[:4])))

        if os.path.isdir(source):
            src, origin, listing = source, "directory", os.listdir(source)
        else:
            from huggingface_hub import HfApi, hf_hub_download
            src, origin, listing, err = None, None, [], None
            for kind in ("model", "dataset"):
                try:
                    p = hf_hub_download(source, "cartridge.json",
                                        repo_type=kind)
                except Exception as e:                  # 404, auth, network
                    err = e
                    continue
                try:
                    listing = HfApi().list_repo_files(source,
                                                      repo_type=kind)
                except Exception:                       # listing is a
                    listing = []                        # courtesy, not a gate
                src, origin = os.path.dirname(p), "%s repo" % kind
                for f in self.CARTRIDGE_FILES[1:]:
                    try:
                        hf_hub_download(source, f, repo_type=kind)
                    except Exception as e:              # log.json optional
                        if f == "state.npz":
                            raise RuntimeError(
                                "%s has a cartridge.json but its "
                                "state.npz could not be fetched (%s)"
                                % (source, e))
                break
            if src is None:
                raise RuntimeError(
                    "no cartridge at %r: it needs a cartridge.json at the "
                    "root of a Hugging Face repo, or a local directory "
                    "written by `sillage export` (%s)" % (source, err))

        man = os.path.join(src, "cartridge.json")
        if not os.path.exists(man):
            raise RuntimeError("%s has no cartridge.json, so it is not a "
                               "sillage cartridge" % src)
        stale = [f for f in listing if f.endswith(".pkl")]
        if stale:
            raise RuntimeError(
                "%s ships %s: a pre-1.5 pickle. Opening one executes "
                "code, and this cartridge is not yours -- refusing. Ask "
                "for it re-exported with sillage >= 1.5."
                % (src, ", ".join(sorted(stale))))
        with open(man, encoding="utf-8") as f:
            manifest = json.load(f)
        # a memory is written in one model's token space. If you pinned a
        # model, say so now rather than after the files are on disk.
        pinned, kw = self._mem_arg
        want = manifest.get("model")
        if want and pinned is not None and want != self.mem.which:
            raise RuntimeError(
                "this cartridge is a %s memory and you asked for --model "
                "%s. A memory is written in one model's token space: drop "
                "--model to open it as %s." % (want, self.mem.which, want))

        os.makedirs(dest, exist_ok=True)
        for f in ("cold.npz", "cold.pkl", "index.json", "index.pkl",
                  "calib.npz", "calib.pkl"):
            p = os.path.join(dest, f)
            if os.path.exists(p) and force:
                os.remove(p)          # a cartridge has none of these, and
        copied = []                   # ours must not answer for its matrices
        for f in self.CARTRIDGE_FILES:
            p = os.path.join(src, f)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(dest, f))
                copied.append(f)
        if want and want != self.mem.which:      # adopt what it declares,
            self.mem = SillageMemory(dest, None, *kw)   # exactly as opening
        else:                                    # this directory would
            self.mem.load()           # this process is holding the old one
        self.index = Index(os.path.join(dest, "index.json"))
        return {"dir": dest, "from": source, "origin": origin,
                "files": copied, "manifest": manifest,
                "bytes": sum(os.path.getsize(os.path.join(dest, f))
                             for f in copied)}

    # ------------------------------------------------------------ review ----
    def review(self):
        """Which documents are about to be forgotten, and why.

        Paper 6 measured the two-occurrence rule: a fact seen once never
        clears the cold store's admission threshold, lives on the
        Hebbian matrix alone, and is gone after ~40k tokens of anything
        else; seen twice, it is still there at +110k. The threshold is
        `COLD_MIN_COUNT`, so the tool can simply *look*: for every
        document it has read, how many of its 4-grams are consolidated
        (count >= 2, served), how many are fragile (count == 1, stored
        but never spoken), and how many are gone (pruned at
        consolidation, or never written).

        Returns one record per source, MOST FRAGILE GRAMS FIRST.
        Ordering by percentage instead was measured to mislead: a
        three-line diary saying "nothing to report" sat at 2% and was
        recommended before a page of decisions at 71%, because a tiny
        document has few grams and one reread carries it to 79%. What is
        about to be forgotten is a COUNT, not a share -- the share is
        reported beside it because it says how far a document has got.
        """
        from .core import COLD_MIN_COUNT, NGRAM
        by_source = {}
        for p in self.index.passages:
            by_source.setdefault(p["source"], []).append(p["text"])
        if not by_source:
            return []
        tok = self.load_tokenizer()
        out = []
        for source, texts in by_source.items():
            ids = tok.encode("\n\n".join(texts))
            solid = weak = gone = 0
            seen = set()
            for i in range(len(ids) - NGRAM + 1):
                gram = np.array(ids[i:i + NGRAM],
                                dtype=np.int32).tobytes()
                if gram in seen:
                    continue
                seen.add(gram)
                slot = self.mem.cold.get(gram)
                if slot is None:
                    gone += 1
                elif sum(slot[1].values()) >= COLD_MIN_COUNT:
                    solid += 1
                else:
                    weak += 1
            total = max(1, solid + weak + gone)
            out.append({"source": source, "grams": total,
                        "consolidated": solid, "fragile": weak,
                        "gone": gone,
                        "share": round(solid / total, 3),
                        "passages": len(texts)})
        return sorted(out, key=lambda r: (-r["fragile"], r["share"]))

    # --------------------------------------------------------------- ask ----
    def ask(self, question, k=3, numeric_only=False):
        """Grounded passages from what has been read. Nothing is generated."""
        return self.index.search(question, k=k, numeric_only=numeric_only)

    def add_to_index(self, path, name=None):
        """Index a document without reading it into the memory (instant).

        `name` overrides the key, exactly as in `read`: without it, two
        `index.md` in two folders of a vault evict each other and a whole
        note disappears without a word. `read` was fixed for this in
        1.8.1; `index` was not, and a trial found it.
        """
        path = os.path.expanduser(path)
        n = self.index.add(read_text(path), name or os.path.basename(path))
        self.index.save()
        return n

    # ------------------------------------------------------------ report ----
    def status(self):
        """Everything `sillage status` prints, as a dictionary."""
        mem = self.mem
        # everything in the directory, not a hand-kept list: watch.json and
        # a pulled cartridge.json live here too, and this figure should
        # equal what `forget --all` would remove
        disk = 0
        if self.state_dir and os.path.isdir(self.state_dir):
            for f in os.listdir(self.state_dir):
                p = os.path.join(self.state_dir, f)
                if os.path.isfile(p):
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
                # a fast read never trains the adapter, and the v2 tier
                # abstains under 500 observations: both look "on" in the
                # size table while contributing nothing
                "adapter_written": bool(mem.fastweights
                                        and float(np.abs(mem.A).max()) > 0),
                "scored_S": len(mem.res_S),
                "writes_per_parameter": mem.writes_per_parameter(),
                "sizes": mem.sizes(), "disk": disk,
                "files": mem.log["files"]}

    def save(self):
        """Consolidate the memory and write both state and index to disk."""
        self.mem.save()
        self.index.save()
