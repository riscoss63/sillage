# Sillage

**Your language model forgets everything. Sillage gives it a memory — and a
way to keep learning — in a fixed handful of megabytes, with no gradients and
no index that grows.**

> *sillage* (n., French) — the trace left behind by something that has passed:
> a ship's wake, a scent in a room. What a model keeps of what it read.

[![tests](https://github.com/riscoss63/sillage/actions/workflows/tests.yml/badge.svg)](https://github.com/riscoss63/sillage/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-green.svg)](requirements.txt)
[![Papers: 4](https://img.shields.io/badge/preprints-4-orange.svg)](papers/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22079016.svg)](https://doi.org/10.5281/zenodo.22079016)

A frozen LM reads your documents, remembers them, and predicts better next
time — **no gradients, no fine-tuning, no growing index**. One Hebbian matrix
written as the model reads, a semantic tier routed by confidence, a cold store
that consolidates by surprise, and a rank-16 adapter on the readout. Four
mechanisms, four papers, **one command-line tool**. Everything runs on a
laptop CPU.

<p align="center"><img src="figs/demo.gif" width="94%" alt="Sillage demo: a frozen LM reads a document on Monday and recalls it on Tuesday"></p>

Two sessions, two days apart, nothing kept in context: on Tuesday the second
draft costs **half** the perplexity it would have cost on Monday (10.68 -> 5.39),
and the model completes a sentence it can only know from what it read.

---

## Quickstart (60 seconds, then a coffee)

```bash
git clone https://github.com/riscoss63/sillage && cd sillage
pip install -e .                      # numpy + torch + transformers
```

```bash
sillage index notes.md                # instant: no model, already queryable
sillage ask "what did the report say?"

sillage read notes.md                 # memorize it (CPU: ~8 min per 10k tokens)
sillage complete "The report said"    # generate WITH the memory
sillage status                        # what it knows, tier by tier
```

```console
$ sillage read preprint_v1.txt                    # Monday, memory empty
read preprint_v1.txt: 5859 tokens in 4.9 min | PPL 12.14 -> 11.83 (adapter) -> 11.71 (+memory)

$ sillage read preprint_v2.md                     # Tuesday, a new process
read preprint_v2.md: 11041 tokens in 9.5 min | PPL 10.68 -> 9.82 (adapter) -> 5.39 (+memory)
memory consolidated and saved (16900 tokens lifetime, 9883 cold grams, 194 passages indexed).
```

Three numbers per file, because there are two mechanisms: what the frozen
model alone predicts, what the rank-16 adapter adds, and what the memory of
everything read so far adds on top.

The memory lives in `./.sillage` and survives restarts — it survived a power
cut during development. Its size is fixed the day you start: 7.4 MB with
GPT-2, 25 MB with Qwen3 (whose vocabulary is 3x larger), the same after one
document or ten thousand. It **never learns from its own generations**,
only from what you give it to read.

Reading is the slow part (one frozen forward pass per token, on CPU): about
8 minutes per 10k tokens with the default Qwen3-0.6B, about 2 with
`--model gpt2` — which is 4x faster but English-only. `index` and `ask` need
no model at all and are instant.

Prefer no install? `pip install -r requirements.txt` and then
`python -m sillage read notes.md` does exactly the same thing — that also
works if the `sillage` command lands outside your PATH.

<details>
<summary><b>Use it from Python (four lines)</b></summary>

```python
from sillage import Sillage

s = Sillage(model="gpt2")          # or "qwen" (Qwen3-0.6B), the default
s.read("notes.md")                 # read, memorize, index -- then save
s.ask("what did the report say?")  # exact passages, nothing generated
print(s.complete("The report said"))
```

`Sillage.status()` returns the same numbers as the CLI as a dict, and
`sillage.SillageMemory` is the mechanism alone (numpy only, no transformers)
if you want to wire it into your own generation loop.
</details>

<details>
<summary><b>Every option</b></summary>

| command | what it does |
|---|---|
| `sillage read FILE...` | read + memorize + index (all four mechanisms) |
| `sillage index FILE...` | index only — instant, no model needed |
| `sillage ask "..."` | grounded excerpts with their source and section |
| `sillage complete "..."` | generate with the memory and the adapter |
| `sillage chat` | both, interactively (`/say`, `/read`, `/status`) |
| `sillage papers` | index the four preprints shipped here, then ask them |
| `sillage demo FILE` | two sessions on one document, start to finish |
| `sillage status` / `forget --all` | inspect / wipe |

Flags: `--model qwen\|gpt2`, `--state DIR` (or `$SILLAGE_STATE`),
`--no-fastweights`, `--no-semantic`, `--half-life N` (forgetting, in tokens),
`-n`, `--temp`, `-k`. Globs are expanded by the tool itself, so
`sillage read docs/*.md` works on Windows too.
</details>

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

### The four papers are the four mechanisms — all of them are in the tool

| paper | mechanism in `sillage/` | on by default | turn it off |
|---|---|---|---|
| 1 · Sillage | `M_G`, 4.2 MB Hebbian *n*-gram tier, amplitude writes, surprise gate | yes | — |
| 2 · Router | `M_S`, 12.6 MB semantic tier, **score-level** mixing with abstention | qwen only (gpt2 needs whitening) | `--no-semantic` |
| 3 · Hierarchy | cold store of exact 4-grams, consolidated by surprise **mass** at save time | yes | — |
| 4 · Fast weights | `A`, rank-16 delta-rule readout adapter, uniform step | yes | `--no-fastweights` |
| 1 & 3 · long horizons | leaky forgetting (half-life in tokens) | no — needed past ~0.5 writes/parameter | `--half-life 100000` |

`sillage status` tells you where you stand on that last line, because the
capacity law is the honest limit of the whole approach.

## The four preprints

All four are archived on Zenodo with permanent DOIs; the LaTeX sources and figures are in [`papers/`](papers/). `sillage papers` indexes them so you can query them offline.

| # | title | the finding |
|---|---|---|
| 1 | **[Sillage](https://doi.org/10.5281/zenodo.22079016)** · [source](papers/sillage/sillage.tex) | a fixed 4.2 MB Hebbian cache beats an unbounded kNN-LM on novel repetitive text |
| 2 | **[Route the Scores, Not the Keys](https://doi.org/10.5281/zenodo.22079444)** · [source](papers/router/router.tex) | gradient-free semantic keys work — but only if you mix at the score level, never in the key |
| 3 | **[One Signal, Three Tiers](https://doi.org/10.5281/zenodo.22079471)** · [source](papers/hierarchy/hierarchy.tex) | consolidating by *surprise mass* keeps 94 % of a cold store's value with 10 % of its entries |
| 4 | **[Memory Remembers, Fast Weights Adapt](https://doi.org/10.5281/zenodo.22079481)** · [source](papers/fastweights/fastweights.tex) | two gradient-free mechanisms, opposite regimes, near-additive gains |

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
`python test_unit.py` checks the four mechanisms themselves in five seconds
(numpy only: retrieval, the square-root rule, forgetting, the delta rule,
consolidation, the state round-trip); `python test_sillage.py` runs 11
end-to-end tests of the tool (each command in its own process, invented facts
the base model cannot know).

<details>
<summary><b>Repository layout</b></summary>

```
sillage/         the tool: core.py (the four mechanisms), runtime.py,
                 index.py (grounded retrieval), cli.py
test_sillage.py  end-to-end tests
papers/          the four preprints (LaTeX + figures)
results/         every number in every paper (JSON)
pipeline/        corpora and frozen-LM passes          \
memory/          the memory systems (papers 1-3)        |  paper
fastweights/     the readout adapter (paper 4)          |  reproduction
eval/            evaluations, controls, diagnostics     |
figures/         figure generation                     /
data/ dumps/     regenerable artifacts (gitignored, ~2 GB)
```

Version 1.0 merged the two former scripts into one tool: `assistant.py` →
`sillage read` / `complete`, `papers_assistant.py` → `sillage papers` / `ask`,
`demo.py` → `sillage demo`. Old `memory_state/` directories are still read.
The research scripts keep their own bootstrap header, so any of them runs from
anywhere and resolves `data/`, `dumps/`, `results/` identically.
</details>

<details>
<summary><b>Caveats worth knowing before you try it</b></summary>

- Tested on small frozen models (GPT-2 124M, Qwen3-0.6B) and CPU-scale streams.
- The memory captures **surface repetition**; paraphrase recall is what the
  semantic tier partially addresses, and it remains the open frontier.
- A fixed matrix saturates at long horizons (~0.5 writes per parameter);
  forgetting (×2.3, `--half-life`) and capacity (×3.4) are the measured
  remedies.
- `sillage forget <file>` removes a document from the index, not from the
  matrices: Hebbian traces are superposed, so only `--all` or forgetting
  removes those. The tool says so rather than pretending otherwise.
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
  doi    = {10.5281/zenodo.22079016},
  url    = {https://doi.org/10.5281/zenodo.22079016}
}
```

MIT licensed. Issues and questions welcome.
