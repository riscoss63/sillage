# Sillage

**Your language model forgets everything. Sillage gives it a memory — and a
way to keep learning — in a fixed handful of megabytes, with no gradients and
no index that grows.**

> *sillage* (n., French) — the trace left behind by something that has passed:
> a ship's wake, a scent in a room. What a model keeps of what it read.

[![PyPI](https://img.shields.io/pypi/v/sillage.svg)](https://pypi.org/project/sillage/)
[![demo](https://img.shields.io/badge/%F0%9F%A4%97-try%20it%20in%20your%20browser-yellow)](https://huggingface.co/spaces/riscoss/Sillage)
[![tests](https://github.com/riscoss63/sillage/actions/workflows/tests.yml/badge.svg)](https://github.com/riscoss63/sillage/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![CPU only](https://img.shields.io/badge/hardware-CPU%20only-green.svg)](requirements.txt)
[![Papers: 5](https://img.shields.io/badge/preprints-5-orange.svg)](papers/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22079016.svg)](https://doi.org/10.5281/zenodo.22079016)

A frozen LM reads your documents, remembers them, and predicts better next
time — **no gradients, no fine-tuning, no growing index**. One Hebbian matrix
written as the model reads, a semantic tier routed by confidence, a cold store
that consolidates by surprise, and a rank-16 adapter on the readout. Four
mechanisms, five papers, **one command-line tool**. Everything runs on a
laptop CPU.

<p align="center"><img src="figs/demo.gif" width="94%" alt="Sillage demo: a frozen LM reads a document on Monday and recalls it on Tuesday"></p>

Two sessions, two days apart, nothing kept in context: on Tuesday the second
draft costs **half** the perplexity it would have cost on Monday (10.68 -> 5.39),
and the model completes a sentence it can only know from what it read.

**[Try it in your browser](https://huggingface.co/spaces/riscoss/Sillage)** —
no install, no account: a memory that has already read one of the papers, and
a box to feed it your own text.

---

## Quickstart (60 seconds, then a coffee)

```bash
pip install sillage                   # numpy + torch + transformers
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

### Which model can it wrap? Any of them

```bash
sillage read notes.md --model qwen                          # shortcut
sillage read notes.md --model HuggingFaceTB/SmolLM2-135M    # any hub id
sillage read notes.md --model ./my-finetuned-llama          # any local folder
```

Nothing here is model-specific: the memory only ever sees next-token logits,
one hidden state and the observed token, so **any causal language model
works** — GPT-2, Qwen, Llama, Mistral, Pythia, SmolLM, your own fine-tune.
Verified across three architectures (GPT-2, Qwen3, GPT-NeoX). Four things to
know before pointing it at a new one:

- **The readout tunes itself on an unfamiliar model.** How strongly the
  memory speaks (`beta`, `lambda`) and when it stays quiet (the abstention
  threshold) were tuned per model in the papers. For any model they did not
  tune, the tool does that tuning itself: one position in three joins a
  rolling window, the published grids are searched on it at the end of each
  read, and the winner governs the next one — so nothing is ever scored with
  settings fitted on itself. For `qwen` and `gpt2` the published settings are
  kept instead, and that is a measured decision, not deference: refitting
  them on your own cold memory *loses* (see below). `--calibrate` forces
  fitting anyway, `--no-calibrate` forbids it, `read --recalibrate` starts
  over.
- The **semantic tier stays off** for an unknown model (`--semantic` to force
  it). Paper 2 is explicit: raw hidden states need whitening except where
  their geometry is already well conditioned, and that has only been verified
  on Qwen3.
- A memory is written in **one model's token space**, so give each model its
  own `--state` directory. After that you can drop `--model`: the state
  remembers which model it belongs to, and refuses to be opened by another.
- **Cost follows the model**: reading time follows its size, and the adapter
  is `vocab x 16` floats (3.2 MB at GPT-2's vocabulary, 9.7 MB at Qwen3's).
  A 7B model on a CPU is possible but slow; this was designed for the 0.1–1B
  class. With `--device cuda` the frozen forward passes move to the GPU while
  the memory stays on the CPU, which is where the mechanisms belong: they are
  rank-1 updates, not matrix multiplications.

What calibration actually does, measured on a model the papers never touched
(Pythia 70M, two documents, second one scored out of sample):

```console
$ sillage read doc_a.txt --model EleutherAI/pythia-70m
read doc_a.txt: 5025 tokens in 1.0 min | PPL 2.02 -> 1.92 (adapter) -> 1.79 (+memory)
fitted the readout on 1675 observations from what was just read (the papers' grids):
  n-gram tier : beta 40, lambda 0.85, abstain below q75
  +0.2461 nats on that window. It governs the NEXT read, never this one,
  so no perplexity printed here was tuned on itself.
```

On the next document that fitted readout gives **1.38** where the GPT-2
defaults give 1.42 — worth having when nobody has tuned your model.

It also tracks the memory as it fills, because each fit sees a warmer memory
than the last. Reading one paper three times (GPT-2, `--calibrate`), where
the readout fitted at the end of a pass is the one that governs the next:

| pass | perplexity of this pass | what its window then fitted |
|---|---|---|
| 1 | 56.0 | `beta 20, lambda 0.1, abstain below q75` |
| 2 | 7.5 | `beta 40, lambda 0.85, abstain below q50` |
| 3 | 1.5 | `beta 40, lambda 0.85, abstain below q25` |

(Frozen GPT-2 alone stays at 68.8 throughout.)

If the `sillage` command lands outside your PATH, `python -m sillage ...`
is identical. For the papers, the reproduction pipeline and the tests, clone
the repository instead:

```bash
git clone https://github.com/riscoss63/sillage && cd sillage && pip install -e .
```

<details>
<summary><b>Use it from Python (four lines)</b></summary>

```python
from sillage import Sillage

s = Sillage(model="gpt2")          # any causal LM; omit it and the state
                                   # says which model it belongs to
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
| `sillage papers` | index the five preprints shipped here, then ask them |
| `sillage demo FILE` | two sessions on one document, start to finish |
| `sillage status` / `forget --all` | inspect / wipe |

Flags: `--model NAME` (see below), `--state DIR` (or `$SILLAGE_STATE`),
`--device cpu|cuda|mps` (a GPU is used for the frozen forward passes when
there is one; the mechanisms stay numpy on the CPU), `--no-fastweights`,
`--no-semantic`, `--half-life N` (forgetting, in tokens), `--no-calibrate` /
`read --recalibrate`, `-n`, `--temp`, `-k`, and on `complete`/`chat`:
`--fast` (speculative decoding from the memory -- identical output, greedy
only) and `--target NAME` (a bigger same-tokenizer sibling reads the state;
adapter off). Globs are expanded by the tool
itself, so `sillage read docs/*.md` works on Windows too.
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
| \+ memory **and** fast weights | **16.8** | **−46 %** | 7.4 MB, constant |

(The last row is the rank-16 adapter the tool actually ships,
[measured](results/fwscale_bhd.json); paper 4's +0.633-nat headline, PPL
16.6, uses the dev-selected rank-256 adapter — 51 MB of adapter for the
last 0.2 nats.)

The fixed 4.2 MB memory beats the unbounded datastore it was designed to
approximate — paired bootstrap **P = 1.000**
([committed](results/paired_bhd_v3_vs_knn.json)), replicated over 5 random
seeds. On a second model (Qwen3-0.6B) it **statistically matches** that
unbounded store at one-sixteenth the memory (paired CI [−0.08, +0.06] nats —
parity, not a win; the internal orderings all replicate). And the gains show
up in behaviour, not just likelihood: recall of recurring technical terms
(word types already seen ≥ 2 times) after one reading pass goes from
**11.3 % to 23.7 %** (McNemar 261:14).

<p align="center"><img src="figs/fig1_main.png" width="88%" alt="Main results"></p>

The same state also pays in **speed**. Used as a drafter for speculative
decoding, it accelerates its own model and -- after two minutes of
read-only calibration -- its bigger tokenizer-siblings, with output
identical to normal greedy decoding: **x1.63 on Qwen3-0.6B, x1.98 on
1.7B, x1.86 on 4B** (T4, 70-87 % of drafted tokens accepted), while on
text it never read the drafter abstains and costs nothing (x0.97-1.09).
And the state transfers: the 0.6B reads the documents once, and the 4B --
which read nothing -- recalls them. Paper 5 measures the whole loop,
controls and negative results included ([`spec/`](spec/),
[results](results/)). The tool ships both: `sillage complete "..." --fast`
(speculative, identical output) and `--target Qwen/Qwen3-1.7B` (a bigger
same-tokenizer sibling reading this state).

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

### Papers 1–4 are the four mechanisms — all of them are in the tool.
### Paper 5 is what the state does next: recall for the family, and speed

| paper | mechanism in `sillage/` | on by default | switch |
|---|---|---|---|
| 1 · Sillage | `M_G`, 4.2 MB Hebbian *n*-gram tier, amplitude writes, surprise gate | yes | — |
| 2 · Router | `M_S`, 12.6 MB semantic tier, **score-level** mixing with abstention | `--model qwen` only (other models need whitening) | `--semantic` / `--no-semantic` |
| 3 · Hierarchy | cold store of exact 4-grams, consolidated by surprise **mass** at save time | yes | — |
| 4 · Fast weights | `A`, rank-16 delta-rule readout adapter, uniform step | yes | `--no-fastweights` |
| 1 & 3 · long horizons | leaky forgetting (half-life in tokens) | no — needed past ~0.5 writes/parameter | `--half-life 100000` |
| all four · protocol | readout **calibration**: the papers' grids fitted on a rolling window of what you just read, governing the next read | only for a model the papers did not tune | `--calibrate` / `--no-calibrate`, `read --recalibrate` |

`sillage status` shows both of the last two lines: which readout is in force
right now, and how close the matrix is to saturation — the capacity law being
the honest limit of the whole approach.

## The five preprints

All five are archived on Zenodo with permanent DOIs; the LaTeX sources and figures are in [`papers/`](papers/). `sillage papers` indexes them so you can query them offline.

| # | title | the finding |
|---|---|---|
| 1 | **[Sillage](https://doi.org/10.5281/zenodo.22079016)** · [source](papers/sillage/sillage.tex) | a fixed 4.2 MB Hebbian cache beats an unbounded kNN-LM on novel repetitive text |
| 2 | **[Route the Scores, Not the Keys](https://doi.org/10.5281/zenodo.22079444)** · [source](papers/router/router.tex) | gradient-free semantic keys work — but only if you mix at the score level, never in the key |
| 3 | **[One Signal, Three Tiers](https://doi.org/10.5281/zenodo.22079471)** · [source](papers/hierarchy/hierarchy.tex) | consolidating by *surprise mass* keeps 92–94 % of a cold store's value with 10 % of its entries (500k streams) |
| 4 | **[Memory Remembers, Fast Weights Adapt](https://doi.org/10.5281/zenodo.22079481)** · [source](papers/fastweights/fastweights.tex) | two gradient-free mechanisms, opposite regimes, near-additive gains |
| 5 | **[The Memory Pays for Itself](https://doi.org/10.5281/zenodo.22109220)** · [source](papers/drafter/drafter.tex) | the same state recalls documents across a model family and speculatively accelerates it (x1.6-2.0, output-identical) |

## What did *not* work (and how we know)

This is the part most repositories leave out.

- **Hidden states make terrible Hebbian keys.** Their similarity geometry is
  too entangled (95th-percentile cosine 0.94 between random pairs,
  [measured](results/semantic_diag_bhd.json)); the best semantic-only
  configuration reached 8 % of the *n*-gram key's gain and was harmful
  off-domain.
- **Surprise gating helps memory and *hurts* fast weights** (−18 % on one
  stream, −100 % on another). The delta rule already carries its own error
  term; gating double-counts it. Gate the mechanisms that cannot see their own
  error; leave alone the ones that can.
- **Two evaluation bugs shipped before any result did.** Raw Gutenberg `\r\n`
  endings inflated every method by **+1.35 phantom nats**; a one-position
  misalignment made the RAG baseline score at chance. Both were caught by
  controls, not by luck.
- **Self-calibration loses to a proper tuning, where one exists.** Fitting the
  readout on your own stream sounds strictly better than using someone else's
  constants. It is not: the window is read by a memory that is *colder* than
  the one those settings will govern, so the fit comes out systematically too
  timid. Measured on GPT-2, fitting on one technical paper and scoring on the
  next (gain over the frozen model, in nats):

  | readout fitted on | gain on the next paper |
  |---|---|
  | the whole window | +0.098 |
  | its recent half | +0.103 |
  | its recent quarter | +0.109 |
  | **the papers' published settings** | **+0.120** |
  | an oracle fitted on that paper itself | +0.125 |

  Recency helps, and never enough. Hence the rule the tool follows: calibrate
  a model nobody has tuned, and keep the published constants for the two that
  were tuned properly — on 36k–500k-token streams, not on a few thousand cold
  ones.

  (Provenance note: this table, the Pythia and three-pass examples above,
  and the quickstart transcripts are replayable CLI sessions of the shipped
  tool, reported as transcripts. Every number in the results tables and the
  five papers is committed as JSON in [`results/`](results/); these
  illustrative CLI numbers are not, and will vary slightly with your
  documents.)

Hence the three controls we now consider mandatory for streaming-memory work —
`python eval/diagnostic.py` runs them: a **shuffled-retrieval null** (replace
retrieved values with random tokens; any surviving gain is an artifact), a
**unigram-cache null**, and a **base-model sanity perplexity**.

---

## Reproducing everything

Every number in every paper regenerates from these scripts with fixed seeds,
on CPU. See **[REPRODUCE.md](REPRODUCE.md)** for the full pipeline; results
are committed as JSON in [`results/`](results/) (including per-seed values).
`python test_unit.py` checks the mechanisms themselves in five seconds
(numpy only, no model: retrieval, the square-root rule, forgetting, the delta
rule, consolidation, the state round-trip, the readout tuner, the multi-model
paths); `python test_sillage.py` runs 13 end-to-end tests of the tool (each
command in its own process, invented facts the base model cannot know).

<details>
<summary><b>Repository layout</b></summary>

```
sillage/         the tool: core.py (the four mechanisms), runtime.py,
                 index.py (grounded retrieval), cli.py
pyproject.toml   packaging: pip install -e . gives you the `sillage` command
test_unit.py     the mechanisms, in five seconds, numpy only
test_sillage.py  the tool, end to end, in its own processes
.github/         CI: the unit tests and a LaTeX check on every push
papers/          the five preprints (LaTeX + figures)
results/         every number in every paper (JSON)
pipeline/        corpora and frozen-LM passes          \
memory/          the memory systems (papers 1-3)        |  paper
fastweights/     the readout adapter (paper 4)          |  reproduction
spec/            the speculative drafter (paper 5)      |
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

- The papers' numbers were measured on two frozen models (GPT-2 124M,
  Qwen3-0.6B) at CPU scale. The tool accepts any causal LM and calibrates
  its readout for it, but nothing here has been measured above 1B parameters,
  and the semantic tier has only been validated on Qwen3.
- The memory captures **surface repetition**; paraphrase recall is what the
  semantic tier partially addresses, and it remains the open frontier.
- A fixed matrix saturates at long horizons (~0.5 writes per parameter);
  forgetting (×2.3, `--half-life`) and capacity (×3.4) are the measured
  remedies.
- `sillage forget <file>` removes a document from the index, not from the
  matrices: Hebbian traces are superposed, so only `--all` or forgetting
  removes those. The tool says so rather than pretending otherwise.
- Parts of the state (`cold.pkl`, `index.pkl`, `calib.pkl`) are Python
  pickles: only open `--state` directories you trust, since unpickling can
  execute code. The main matrices (`state.npz`) load with
  `allow_pickle=False`.
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
