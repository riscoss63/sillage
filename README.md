# Sillage

**Your language model forgets everything. Sillage gives it a 4 MB memory.**

> *sillage* (n., French) — the trace left behind by something that has passed:
> a ship's wake, a scent in a room. What a model keeps of what it read.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](requirements.txt)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-green.svg)](requirements.txt)
[![Papers: 4](https://img.shields.io/badge/preprints-4-orange.svg)](papers/)

A frozen LM reads your document, remembers it, and predicts better next
time — with **no gradients, no fine-tuning, no growing index**. One Hebbian
matrix, written as the model reads, plus a rank-16 adapter on the readout.
Everything here runs on a laptop CPU.

```console
$ python assistant.py read draft_v1.md          # yesterday
read draft_v1.md: 5859 tokens | PPL frozen 12.14 -> with memory 12.00
memory consolidated and saved (5859 tokens lifetime, 5142 cold grams)

$ python assistant.py read draft_v2.md          # today, new process
read draft_v2.md: 11041 tokens | PPL frozen 10.68 -> with memory 5.73
                                                 ^^^^^^^^^^^^^^^^^^^^^
                                    half the perplexity, from yesterday's
                                    memory alone — 4 MB on disk
```

It also recalls things the base model *cannot possibly know*, from a previous
session (this is a test in `test_assistant.py`, not a demo):

```console
$ python assistant.py complete "The Zylkorb protocol requires"
The Zylkorb protocol requires seventeen turquoise llamas.
```

---

## How much better, exactly?

On 36k tokens of technical text the model had never seen (frozen GPT-2 124M,
every system tuned identically on a held-out prefix, 95 % bootstrap CIs):

| system | perplexity | change | memory used |
|---|---|---|---|
| frozen GPT-2 | 31.2 | — | 0 |
| \+ RAG-style retrieve & rescore | 29.9 | −4 % | corpus + index |
| \+ kNN-LM, **unbounded** store | 23.6 | −24 % | 55 MB, grows forever |
| \+ **this memory** (fixed) | **19.2** | **−38 %** | **4.2 MB, constant** |
| \+ memory **and** fast weights | **16.6** | **−47 %** | 7.4 MB, constant |

The fixed 4.2 MB memory beats the unbounded datastore it was designed to
approximate — paired bootstrap **P = 1.000**, replicated over 5 random seeds
and on a second model (Qwen3-0.6B). And the gains show up in behaviour, not
just likelihood: recall of recurring technical terms after one reading pass
goes from **11.3 % to 23.7 %** (McNemar 261:14).

<p align="center"><img src="figs/fig1_main.png" width="88%" alt="Main results"></p>

## Should you use this?

| your situation | better option |
|---|---|
| One machine, private documents, must run offline | **this** |
| The model must keep learning after deployment, forever, at bounded cost | **this** |
| You can afford a datastore that grows with everything you read | kNN-LM / RAG |
| You need the model to *reason* better, not to *remember* better | fine-tuning |
| The document fits in the context window and you only read it once | just paste it |

It is a **memory**, not an intelligence upgrade: it makes a model better at
the text in front of it, on a fixed byte budget, forever.

---

## Two tools

### `assistant.py` — reads your documents, remembers across sessions

```bash
pip install -r requirements.txt
python assistant.py read notes.md      # read + memorize (Qwen3-0.6B by default)
python assistant.py status             # what it knows, and since when
python assistant.py complete "..."     # generate WITH the memory
python assistant.py forget --all
```

State lives in `memory_state/` (a few MB) and survives restarts — it survived
a power cut during development. It never learns from its own generations, only
from what you give it to read. `python test_assistant.py` runs 7 end-to-end
tests (each session a separate process, invented facts the base model cannot
know); all pass.

### `papers_assistant.py` — the four preprints, queryable offline

```bash
python papers_assistant.py build                    # index the papers (instant)
python papers_assistant.py ask "does rank 16 suffice?"
python papers_assistant.py chat                     # interactive
```

```console
[1] FastWeights · Rank 16 is enough   (relevance 0.187)
    At r = 16 the adapter is 3.2 MB on GPT-2 and 9.7 MB on Qwen3 - the same
    budget class as the memory it complements, and 16x smaller than the
    r = 256 adapter with which we began.
```

`ask` only ever returns **real passages with their paper and section** —
nothing is generated, so there is no hallucination surface. `say` is the
opposite (a 0.1–0.6B model writing prose with the papers in memory): useful
for watching the memory work, never as a source of truth. The tool says so
itself.

---

## How it works, in four ideas

<p align="center"><img src="figs/fig0_architecture.png" width="82%" alt="architecture"></p>

1. **Keys by binding.** Each position is addressed by a sliding *n*-gram
   product of random token hypervectors — a key that fires on exact
   repetition and stays near-orthogonal otherwise.
2. **Values as amplitudes.** The matrix stores *square roots* of accumulated
   mass, not counts. That one change roughly doubles the memory's benefit:
   square-root compression stops frequent continuations from drowning rare
   ones in superposition.
3. **Surprise decides.** Every write is scaled by the model's own token
   surprise `−ln p_LM` — free at inference. The same scalar also arbitrates
   which tier answers, and which patterns get consolidated to cold storage.
4. **Fast weights adapt what memory cannot.** A rank-16 delta-rule adapter on
   the readout (3.2 MB, no error transported through any layer) wins exactly
   where memory is weakest, and their gains add up (89–98 % additive).

## The four preprints

All in [`papers/`](papers/), with figures and [submission notes](papers/SUBMISSION.md).

| # | title | the finding |
|---|---|---|
| 1 | [Surprise-Gated Amplitude Memory](papers/sillage/sillage.tex) | a fixed 4.2 MB Hebbian cache beats an unbounded kNN-LM on novel repetitive text |
| 2 | [Route the Scores, Not the Keys](papers/router/router.tex) | gradient-free semantic keys work — but only if you mix at the score level, never in the key |
| 3 | [One Signal, Three Tiers](papers/hierarchy/hierarchy.tex) | consolidating by *surprise mass* keeps 94 % of a cold store's value with 10 % of its entries |
| 4 | [Memory Remembers, Fast Weights Adapt](papers/fastweights/fastweights.tex) | two gradient-free mechanisms, opposite regimes, near-additive gains |

## What did *not* work (and how we know)

This is the part most repositories leave out.

- **Hidden states make terrible Hebbian keys.** Their similarity geometry is
  too entangled (95th-percentile cosine 0.93 between random pairs); the best
  semantic-only configuration reached 8 % of the *n*-gram key's gain and was
  harmful off-domain.
- **Surprise gating helps memory and *hurts* fast weights** (−18 % on one
  stream, −100 % on another). The delta rule already carries its own error
  term; gating double-counts it. Gate the mechanisms that cannot see their own
  error; leave alone the ones that can.
- **Two evaluation bugs shipped before any result did.** Raw Gutenberg `\r\n`
  endings inflated every method by **+1.35 phantom nats**; a one-position
  misalignment made the RAG baseline score at chance. Both were caught by
  controls, not by luck.

Hence the three controls we now consider mandatory for streaming-memory work —
`python eval/diagnostic.py` runs them: a **shuffled-retrieval null** (replace
retrieved values with random tokens; any surviving gain is an artifact), a
**unigram-cache null**, and a **base-model sanity perplexity**.

---

## Reproducing everything

Every number in every paper regenerates from these scripts with fixed seeds,
on CPU. See **[REPRODUCE.md](REPRODUCE.md)** for the full pipeline; results
are committed as JSON in [`results/`](results/) (including per-seed values).

<details>
<summary><b>Repository layout</b></summary>

```
assistant.py  papers_assistant.py  demo.py  test_assistant.py   the tools
pipeline/     corpora and frozen-LM passes
memory/       the memory systems (papers 1-3)
fastweights/  the readout adapter (paper 4)
eval/         evaluations, controls, diagnostics
figures/      figure generation
papers/       the four preprints (LaTeX + figures)
results/      every number in every paper (JSON)
data/ dumps/  regenerable artifacts (gitignored, ~2 GB)
```

Any script runs from anywhere: a small bootstrap header finds the repo root,
fixes `sys.path` and `chdir`s, so relative paths always resolve.
</details>

<details>
<summary><b>Caveats worth knowing before you try it</b></summary>

- Tested on small frozen models (GPT-2 124M, Qwen3-0.6B) and CPU-scale streams.
- The memory captures **surface repetition**; paraphrase recall is what the
  semantic tier partially addresses, and it remains the open frontier.
- A fixed matrix saturates at long horizons (~0.5 writes per parameter);
  forgetting (×2.3) and capacity (×3.4) are the measured remedies.
- The papers' *Manuscripts* stream (unpublished drafts) is not redistributed —
  drop your own documents in `manuscripts/` to run that protocol.
</details>

## Citation

```bibtex
@article{sghairi2026sillage,
  title  = {Sillage: Surprise-Gated Amplitude Memory
            for Frozen Language Models},
  author = {Sghairi, Abderrahmane},
  year   = {2026},
  note   = {Preprint. arXiv identifier to be added.}
}
```

MIT licensed. Issues and questions welcome.
