"""Tokenize the three short-domain corpora with the Qwen3 tokenizer
(same cleaned texts as data_prep.py). Outputs data/q_<domain>_ids.npy.

DECLARED CONFOUND (kept as-is so the published Qwen results reproduce
exactly): the 40k cap is in TOKENS, and Qwen3's tokenizer is denser than
GPT-2's, so on the Gutenberg classics the two models do not read the same
span of text -- Qwen reads all of Alice (35,272 tokens for the full 145k
chars, where GPT-2's 40k tokens cover only a truncation) and a longer slice
of Einstein. Cross-MODEL comparisons on those two streams therefore compare
different content; within-model orderings are unaffected, and the
Manuscripts stream is clean (both models read the whole text). meta.json
records chars_total vs chars_used per stream so the mismatch is visible.
A content-matched re-prep (decode GPT-2's truncated ids, re-tokenize that
text with Qwen) is the fix if cross-model deltas ever become the claim."""


# --- repo bootstrap: run this script from anywhere ---
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

import json
import os

import numpy as np

from data_prep import (BHD_FILES, GUTENBERG, fetch, normalize_newlines,
                       strip_gutenberg, unwrap_paragraphs)

MAX_TOKENS = 40_000


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    texts = {}
    for name, urls in GUTENBERG.items():
        texts[name] = unwrap_paragraphs(
            normalize_newlines(strip_gutenberg(fetch(urls))))
    if BHD_FILES:            # your own documents in ./manuscripts/, if any
        texts["bhd"] = normalize_newlines(
            "\n\n".join(open(p, encoding="utf-8").read() for p in BHD_FILES))
    else:
        print("NOTE: ./manuscripts/ is empty -- skipping the 'bhd' stream.")
    meta_path = os.path.join("data", "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name, text in texts.items():
        ids = np.array(tok.encode(text), dtype=np.int32)[:MAX_TOKENS]
        np.save(os.path.join("data", f"q_{name}_ids.npy"), ids)
        # chars_used = the span the truncated ids actually cover, so the
        # cross-model content mismatch (see module docstring) stays visible
        chars_used = (len(text) if len(tok.encode(text)) <= MAX_TOKENS
                      else len(tok.decode(ids)))
        meta[f"q_{name}"] = {"chars_total": len(text),
                             "chars_used": int(chars_used),
                             "tokens": int(len(ids))}
        print(f"q_{name}: {len(text)} chars ({chars_used} used) -> "
              f"{len(ids)} tokens", flush=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
