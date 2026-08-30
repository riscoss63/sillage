# Reproducing every number in the eight papers

This file is for the *research* scripts. If you only want to use the system,
`pip install -e .` and `sillage read yourfile.md` is the whole story
(see [README.md](README.md)); `python test_unit.py` checks the mechanisms
themselves in five seconds (21 checks), `python test_sillage.py` runs 14
end-to-end tests of the tool in about twenty minutes, `python test_serve.py`
starts the HTTP service and talks to it over real sockets (16 checks), and
`python test_axis4.py` covers watch, review, export and pull, and the same
commands through the command line (28 checks).

Sections 0–4 (papers 1–4) run entirely on CPU with fixed seeds. Total: roughly
2 h for the corpora and frozen-LM passes, then 3–6 h for the experiments (most
of them can be run selectively — each block below is independent once the dumps
exist). Section 5 builds the figures of all eight papers, and is instant once
the results exist. The papers 5–8 sections at the end of this file need their
own runs, and papers 5, 6 and 7 have GPU arms (a T4 is enough); their CPU arms
are marked as such.

Any script can be launched from anywhere, but by two different mechanisms. The
papers 1–4 pipeline (`pipeline/`, `memory/`, `eval/`, `fastweights/` and
`figures/make_figures{,_p2,_p3,_p4}.py`) carries a bootstrap header that
resolves the repo root and `chdir`s to it, so `data/`, `dumps/`, `results/` and
`papers/*/figs` always resolve identically. The `spec/`, `behav/` and
`longmemeval/` scripts and figures p5–p8 do not: they build absolute paths from
their own directory instead. Figures p5–p8 still write into `papers/*/figs`,
but the experiment scripts write into `spec/results/`, `behav/results/` and
`longmemeval/results/`, all three gitignored, and `spec/bench_gpu.py` writes
its `results_gpu_*.json` into the current directory.

That difference matters when you compare. The papers 1–4 scripts write into
`results/` directly, so their outputs are what you diff against. The `spec/`,
`behav/` and `longmemeval/` files under `results/` are promoted copies,
renamed on promotion: the `behav/` behavioral-law outputs gain a `behav_`
prefix (`behav/results/retention_gpt2.json` ->
`results/behav_retention_gpt2.json`, and likewise `cold_retention_`,
`two_occurrences_`, `equivalence_`, `adversarial_`); `longmemeval/` keeps its
`lme_*` names; `spec/` becomes `drafter_*` (`calib_17b.json` ->
`drafter_calib_17b.json`, `bench_gpt2_papers.json` -> `drafter_cpu_gpt2.json`,
`bench_qwen_cross.json` -> `drafter_cpu_cross.json`, `results_gpu_*.json` ->
`drafter_gpu_*.json`, and the microbenchmark as `drafter_micro_t4.json`); and
the `behav/` probe outputs (`semantic_*`, `shareable_*`, `quantised_*`,
`ship_readout_*`) keep theirs. Diff your own run against the promoted copy,
not against a file your run never wrote.

```bash
pip install -r requirements.txt
```

## 0. Corpora and frozen-LM passes (needed by everything)

```bash
python pipeline/data_prep.py            # 3 short streams (network, once)
python pipeline/data_prep_500k.py       # War and Peace, King James Bible
python pipeline/data_prep_qwen.py       # same texts, Qwen3 tokenizer
python pipeline/dump_any.py gpt2 bhd relativity alice tolstoy
python pipeline/dump_500k.py
python pipeline/dump_any.py qwen bhd relativity alice
```

One pass per stream produces hidden states, base log-probabilities and the
base model's top-128 candidates. **Every method downstream consumes exactly
these dumps**, so no baseline ever sees more than another.

The `bhd` ("Manuscripts") stream is built from the author's unpublished
drafts, which are not redistributed. Put your own novel `.txt`/`.md` documents
in `manuscripts/` to run that protocol with your own contamination-free
domain; the public streams work as-is.

## 1. Memory — baselines, the factorial, replications (paper 1)

```bash
python memory/memories.py                    # kNN-LM, capped kNN, nulls,
                                             # hidden-state keys (negative)
python memory/ngram_memory.py 0.0 1.0        # n-gram memory + exact dictionary
python eval/rag_baseline.py                  # RAG-style retrieve & rescore
python memory/sillage_factorial.py           # amplitudes x gates x key scales
python memory/multiseed.py bhd               # 5-seed replication
python memory/exp_500k.py tolstoy && python memory/exp_500k.py bible
python memory/exp_500k_bigD.py               # capacity sweep
python memory/model2_qwen.py bhd             # second model
python eval/cloze_eval.py gpt2 bhd           # downstream recall
python eval/paired_test_v3.py                # headline paired test
python eval/qwen_beta_ext.py                 # appendix grid-edge check
                                             # (beta extended to 320/640)
```

## 2. Semantic keys and routing (paper 2)

```bash
python eval/semantic_diag.py gpt2 alice          # geometry + LSH diagnostics
python eval/semantic_diag.py gpt2 bhd            # paper 1's p95 = 0.94 datum
python memory/sillage_semantic.py tune gpt2 alice   # key-level hybrid (fails)
python memory/sillage_router.py gpt2 bhd 999999999 multi
python memory/multiseed_router.py gpt2 bhd
python memory/router_500k.py tolstoy decay
python eval/cloze_router.py gpt2 bhd
```

Run the diagnostics first: they are what justified the design, and they take
minutes.

## 3. Hierarchy and consolidation (paper 3)

```bash
python memory/hierarchy_500k.py bible     # instruments one pass, caches it
python memory/hierarchy_500k.py tolstoy
python eval/hier_diag.py tolstoy          # coherence check vs paper 2
```

The instrumented pass is cached in `results/hier_cache_*.npz`; re-running the
script re-evaluates every admission policy and capacity from the cache in
seconds instead of hours.

## 4. Fast weights (paper 4)

```bash
python fastweights/fastweights.py gpt2 bhd            # learning-rate sweep
python fastweights/fastweights_combo.py bhd           # rank x gating + memory
python fastweights/fastweights_scale.py gpt2 tolstoy 16      # 500k stability
python fastweights/fastweights_scale.py qwen bhd 16,64       # second model
```

## 5. Figures

```bash
python figures/make_figures.py      # paper 1  -> papers/sillage/figs
python figures/make_figures_p2.py   # paper 2  -> papers/router/figs
python figures/make_figures_p3.py   # paper 3  -> papers/hierarchy/figs
python figures/make_figures_p4.py   # paper 4  -> papers/fastweights/figs
python figures/make_figures_p5.py   # paper 5  -> papers/drafter/figs
python figures/make_figures_p6.py   # paper 6  -> papers/behavior/figs
python figures/make_figures_p7.py   # paper 7  -> papers/benchmark/figs
python figures/make_figures_p8.py   # paper 8  -> papers/paraphrase/figs
```

## Integrity controls

```bash
python eval/diagnostic.py       # shuffled-retrieval null, unigram null,
                                # base-model sanity perplexity
python eval/smoke_test.py       # end-to-end pipeline check on synthetic data
```

Run these before trusting any new result, including your own: they caught a
corpus-formatting artifact worth +1.35 phantom nats and a one-position
misalignment in a baseline.

## Statistical conventions

Identical across papers 1–4: development split = first 20 % of each
stream (hyperparameters tuned there, then frozen), test = remaining 80 %;
95 % block-bootstrap confidence intervals over 512-token blocks; headline
comparisons use *paired* bootstraps on identical positions; multi-seed studies
re-tune per seed and report mean ± SEM.

The shipped tool follows the same protocol when it meets a model these papers
did not tune: a rolling window of what it has just read is its development
split, the same grids are searched on it (`BETAS`, `LAMS`, `THRESH_Q` in
`sillage/core.py`, taken from `memory/memories.py`, plus the beta = 320 that
paper 1's appendix checked at the grid edge), and the winner governs
the next read -- never the read it was fitted on. For GPT-2 and Qwen3 it keeps
the settings tuned here instead, because those were fitted on 36k-500k-token
streams and a window read by a cold memory measurably loses to them (the
comparison is in the README, under what did not work).

Paper 5 (the speculative drafter) has its own pipeline under `spec/`:
the numpy reference engine and the GPU-resident engine (cross-checked
token-for-token), the CPU campaigns, `bench_gpu.py` with `--calibrate`,
and `spec/kaggle/make_kit.py` to assemble the T4 kit. Its numbers are the
`drafter_*.json` files in `results/`.

Paper 6 (the behavioral laws) lives under `behav/`: the six-probe suite
(`behavioral.py`), retention and its four-voice decomposition, the
two-occurrence rule, the context-equivalence scorer, the adversarial
arms with their defense probes and the `amp_write` instrumentation --
plus `behav/JOURNAL.md`, the lab log where every experiment's
predictions were registered before it ran. Its numbers are the
`behav_*.json` files in `results/` (v2 adds `behav/kaggle_4b/`, the capacity
kernel, and `results/behav_4b.json`). States are never shipped -- a cold
store reveals what it read -- and every script says how to rebuild them.

Two of that paper's outputs need a word. `equivalence.py` scores the
recitation regime by default and the transfer regime with
`--doc path/to/an/unread/sibling.md`; the document behind the committed
`behav_equivalence_gpt2_transfer.json` is one of the author's manuscripts
and is not redistributed, so point `--doc` at a sibling of your own. And
the write-level trace behind the adversarial section is the one behavioral
output *not* committed: `python behav/adversarial_probes.py trace`
regenerates it into `behav/results/`, and the numbers quoted from it are in
`behav/JOURNAL.md`.

Paper 7 (LongMemEval) lives under `longmemeval/`: the extractive arm
over all 500 questions (`arm_e_extractive.py`, CPU-only), the fast
ingestion module with its bit-exact and blocked modes and their
four-state equivalence test (`fast_ingest.py`,
`test_fast_ingest.py`), and the self-contained arm-G kernel under
`longmemeval/kaggle/`. The benchmark data itself is downloaded from
its authors (`xiaowu0162/longmemeval` on the Hugging Face hub) into
`longmemeval/data/` and is never redistributed here. Its numbers are
the `lme_*.json` files in `results/`.

Paper 8 (the paraphrase wall) is thirteen `behav/probe_*.py` scripts,
enumerated in execution order in the paper's reproduction appendix
(`papers/paraphrase/paraphrase.tex`), from `probe_semantic_diag.py`
to `probe_gpt2_zca.py` -- the refutation staircase, the layer sweep,
the post-fix matrix, the query pooling, the behavioral conversion,
and the GPT-2 replication with its ZCA arm. The other `probe_*.py`
files under `behav/` are the shipping probes, below. Its numbers are
the thirteen `semantic_*.json` files that appendix lists (the
`semantic_diag_{alice,bhd,tolstoy,q_alice}.json` files in the same
directory are section 2's, from `eval/semantic_diag.py`); every
prediction, both refuted theories and the gate off-by-one incident
are in `behav/JOURNAL.md`, written before each run.

## Shipping probes

Eleven probes under `behav/` measure what the *shipped* tool does rather than
what a paper claims; the README's `--dtype` table is one of their outputs.
Three of them write into `behav/results/` like the rest of `behav/`, and the
committed copies under `results/` keep the same names; the fourth
(`probe_ship_threshold.py`) writes nothing and prints its sweep. Their
verdicts, and the
predictions registered before each run, are in `behav/JOURNAL.md`.

```bash
python behav/probe_shareable_state.py [gpt2|qwen]
python behav/probe_quantised_gate.py [qwen|gpt2] [int8|bfloat16]
python behav/probe_ship_readout.py [gpt2|qwen]
python behav/probe_ship_threshold.py [qwen|gpt2]
python behav/probe_heading_index.py
python behav/probe_ask_french.py
python behav/probe_serve_midread.py [--model gpt2]
python behav/probe_ask_abstain.py
python behav/probe_ask_stem.py
python behav/probe_ask_abstain.py
python behav/probe_ask_ranking.py

# why `complete` invents: four hypotheses, three refuted (1.9.0)
python behav/probe_readout_dial.py [--target Qwen/Qwen3-1.7B]
python behav/probe_linewrap.py
python behav/probe_outvoted.py
python behav/probe_tokenkey.py
python behav/probe_reflow.py
python behav/probe_moredocs.py
python behav/probe_freeze_mu.py
python behav/probe_abstain_gen.py
python behav/probe_crosscheck.py
```

**The 1.9.0 series answers one question**: `complete` recalls a fact
perfectly from one phrasing and fabricates from another -- why. Run them
in the order above; each was registered with its falsification threshold
before it ran, and three of the four hypotheses died.

`probe_readout_dial.py` tests the readout constants at two capacities
(`--target` for the second). `probe_linewrap.py` and `probe_outvoted.py`
are the two refutations: line wrapping alone does not explain the loss,
and the missing fact is *absent* from the cold store rather than
outvoted in it. `probe_tokenkey.py` isolates the cause -- the same fact,
three phrasings, and the key `[' responsable', ',
', 'mad', 'ame']`
becomes `[' responsable', ',', ' mad', 'ame']` when the line is
rejoined. `probe_reflow.py` measures the fix (7/8 -> 8/8).

`probe_moredocs.py` answers the scaling worry it raised: reading three
unrelated documents afterwards leaves recall at 8/8 throughout, so the
running centre drifting is not a problem in normal use --
`probe_freeze_mu.py` shows the generation-time freeze is hygiene (the
eight answers are byte-identical), not a gain.

`probe_abstain_gen.py` and `probe_crosscheck.py` are the pair worth
reading last. The first tests, on a corpus that set nothing, whether the
memory's own contribution says when it does not know: 7 of 8 unanswerable
questions are correctly refused, and both reworded questions it answers
are right. The second tries three ways to catch the one that slips
through and **fails at all three** -- see the twin questions in
`results/crosscheck.json`, where the identical completion is the correct
answer to one question and a fabrication for the other, with every
signal the same.

`probe_shareable_state.py` asks what a state is worth without its cold store
and its index -- the two tiers a shared cartridge cannot carry, because both
hold text in the clear. It cuts them at *generation* time on one state, so
nothing is rewritten and the comparison is paired
(`results/shareable_gpt2.json`, `results/shareable_qwen.json`).

`probe_quantised_gate.py` reads the same document in two precisions and
compares what matters: the surprise gates themselves, the cold-store
admissions (the two-occurrence rule) and the recall. Its thresholds are
declared in its docstring before the run -- gate correlation > 0.98,
admissions equal to within 1 %, failing which the quantised mode is announced
as approximate. The two committed runs are
`results/quantised_qwen_int8.json` and `results/quantised_qwen_bfloat16.json`.

`probe_ship_readout.py` re-measures beta and lambda for the v2 tier in the
shipped path: a SimHash tier's scores are flatter than the dense-key
prototype's, so `softmax(beta*s)` does not peak the same way. Grid on 10 dev
facts, reported on 10 test facts the grid never saw, locality on 10 witnesses
(`results/ship_readout_gpt2.json`).

`probe_ship_threshold.py` sweeps the quantile of the in-document null that
sets the abstention threshold, and reports paraphrase recall against locality
(witnesses whose greedy completion changes) at each one. Declared criterion:
the smallest quantile whose locality stays at or below 1/10. It writes no
JSON -- the sweep is printed, and the verdict is in `behav/JOURNAL.md`.

`probe_heading_index.py` was written after using the tool rather than after
reading it: on a notebook whose subject sits in the section headings, a
question naming that subject found nothing, because `Index._rebuild`
tokenised the passage text alone. It scores eleven questions against the
same notebook with and without heading and filename tokens
(`results/heading_index.json`): heading-only questions go from **1/5 to
5/5**, the six that already worked stay at 6/6, and no body answer loses
its rank. Needs no model.

`probe_ask_french.py` is the same notebook, wider: 23 questions in four
groups -- the subject in the heading, the subject in the body, the same
questions typed *without accents* as people actually type them, and six
questions the notebook cannot answer, where the only right reply is
silence. It found two more defects (`results/ask_french.json`). The STOP
list is written unaccented while French is not, so `etait`, `meme` and
`apres` carried full idf and any question containing "c'etait" could be
answered by any passage containing "etait"; and nothing floored the score,
so one shared word of the filename matched every passage of a document.
Folding accents in the search key took unaccented questions from 4/6 to
6/6 -- but on its own it did **not** restore silence, which is a
registered prediction refuted in the file. The floor is what does: the
lowest genuine hit scores 0.161, the highest accidental one 0.024, and
`MIN_SCORE` sits between them. Needs no model.

`probe_ask_abstain.py` is the sequel that refuted `probe_ask_french.py`'s
own recommendation. It scores both notebooks -- the tuned one and the
larger one the real-world trials wrote, which it never tunes on -- and
found that the 0.05 floor shipped in 1.8.2 could not survive a second
corpus (genuine hits 0.127-0.285 against accidental 0.133-0.261, fully
overlapping) and had taken a real answer away. Its own finding: the
French elisions (`qu'`, `jusqu'`, `lorsqu'`) survived tokenisation as
full-weight words, and one of them carried 100 % of a false positive's
score. The floor is gone; the elisions and the interrogatives joined the
stop list; a low top score is now *reported* rather than filtered
(`results/ask_abstain.json`).

`probe_ask_stem.py` is the eighth negative result of the series. A light
French suffix stripper, measured on the same two notebooks, answered
FEWER morphological questions than the plain tokeniser (1 of 4 against
2 of 4) because stripping moves a query term and a passage term
independently. Not shipped (`results/ask_stem.json`).

`probe_serve_midread.py` times a chat completion sent *during* an
ingestion, against the same completion sent to an idle server
(`results/serve_midread.json`). It exists because the 1.6.0 release
claimed "a reply in 3.3 s during an ingestion" and a real-world trial
measured 113 s: the lock was handed back only at 1024-token window
boundaries, so a document that fitted in one window had no yield point at
all. Yielding every 32 tokens brings it to **4.35 s at worst against a
1.90 s idle baseline**. Runs on gpt2 in under a minute.
