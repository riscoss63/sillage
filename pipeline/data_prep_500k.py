"""Prepare the two 500k-token long streams.

  tolstoy : War and Peace (Gutenberg #2600)  - long coherent narrative,
            the hard low-repetition case
  bible   : King James Bible (Gutenberg #10) - long formulaic/repetitive,
            the memory-friendly case

Same cleaning as the 40k corpora (normalize newlines, unwrap paragraphs).
Outputs: data/<name>_ids.npy (int32, <= 500k tokens), updates data/meta.json
"""


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

from data_prep import fetch, normalize_newlines, strip_gutenberg, \
    unwrap_paragraphs

MAX_TOKENS = 500_000
BOOKS = {
    "tolstoy": ["https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
                "https://www.gutenberg.org/files/2600/2600-0.txt"],
    "bible": ["https://www.gutenberg.org/cache/epub/10/pg10.txt",
              "https://www.gutenberg.org/files/10/10-0.txt"],
}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    meta_path = os.path.join("data", "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name, urls in BOOKS.items():
        text = unwrap_paragraphs(normalize_newlines(
            strip_gutenberg(fetch(urls))))
        ids = np.array(tok.encode(text), dtype=np.int32)[:MAX_TOKENS]
        np.save(os.path.join("data", f"{name}_ids.npy"), ids)
        meta[name] = {"chars": len(text), "tokens": int(len(ids))}
        print(f"{name}: {len(text)} chars -> {len(ids)} tokens", flush=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
