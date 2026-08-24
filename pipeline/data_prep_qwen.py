"""Tokenize the three short-domain corpora with the Qwen3 tokenizer
(same cleaned texts as data_prep.py). Outputs data/q_<domain>_ids.npy."""


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
    texts["bhd"] = normalize_newlines(
        "\n\n".join(open(p, encoding="utf-8").read() for p in BHD_FILES))
    meta_path = os.path.join("data", "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name, text in texts.items():
        ids = np.array(tok.encode(text), dtype=np.int32)[:MAX_TOKENS]
        np.save(os.path.join("data", f"q_{name}_ids.npy"), ids)
        meta[f"q_{name}"] = {"chars": len(text), "tokens": int(len(ids))}
        print(f"q_{name}: {len(text)} chars -> {len(ids)} tokens", flush=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
