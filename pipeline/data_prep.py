"""Prepare the three domain corpora and tokenize them with GPT-2 BPE.

Domains:
  relativity : Einstein, "Relativity: The Special and General Theory"
               (Project Gutenberg #30155) - technical, repeated terminology
  alice      : "Alice's Adventures in Wonderland" (Gutenberg #11) - narrative
  bhd        : the three local BHD manuscripts concatenated - guaranteed
               post-training-cutoff novel domain with heavy repeated structure

Outputs: data/<domain>_ids.npy (int32 token ids), data/meta.json
Also builds the continual stream: aba = relativity[:15k] + alice[:15k]
+ relativity[15k:30k].
"""


# --- repo bootstrap (added by reorganize.py) ---
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
import re
import urllib.request

import numpy as np

MAX_TOKENS = 40_000
DATA = "data"
GUTENBERG = {
    "relativity": [
        "https://www.gutenberg.org/cache/epub/30155/pg30155.txt",
        "https://www.gutenberg.org/files/30155/30155-0.txt",
    ],
    "alice": [
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "https://www.gutenberg.org/files/11/11-0.txt",
    ],
}
# The "Manuscripts" stream: novel technical documents unseen by the base
# models. The paper uses three drafts of the author's unpublished manuscript
# (not redistributed). To reproduce the *protocol* with your own novel
# domain, drop any .txt/.md files into a local `manuscripts/` folder.
import glob as _glob

BHD_FILES = sorted(_glob.glob("manuscripts/*"))
if not BHD_FILES:
    BHD_FILES = [
        r"C:\Users\abdel\Documents\preprint\preprint_bhd_active_inference.txt",
        r"C:\Users\abdel\Documents\preprint\PREPRINT_V2_1_BHD_ACTIVE_INFERENCE.md",
        r"C:\Users\abdel\Documents\preprint\PREPRINT_V2_2_BHD_ACTIVE_INFERENCE.md",
    ]


def fetch(urls):
    last = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not fetch any of {urls}: {last}")


def strip_gutenberg(text):
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text.strip()


def normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def unwrap_paragraphs(text):
    """Undo Gutenberg hard wrapping: join lines within a paragraph, keep
    paragraph breaks. Without this, GPT-2 faces \\r / fixed-width line breaks
    it essentially never predicts, and any cache 'wins' by predicting
    formatting instead of language (verified by our shuffle control)."""
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        lines = [ln.strip() for ln in p.split("\n")]
        joined = " ".join(ln for ln in lines if ln)
        if joined:
            out.append(joined)
    return "\n\n".join(out)


def main():
    os.makedirs(DATA, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

    texts = {}
    for name, urls in GUTENBERG.items():
        texts[name] = unwrap_paragraphs(
            normalize_newlines(strip_gutenberg(fetch(urls))))
    # BHD manuscripts: line structure (tables, headers) is part of the
    # domain and stays; only line endings are normalized.
    try:
        texts["bhd"] = normalize_newlines(
            "\n\n".join(open(p, encoding="utf-8").read() for p in BHD_FILES))
    except FileNotFoundError:
        print("NOTE: no manuscripts found — the 'bhd' (Manuscripts) stream "
              "is skipped.\nPut your own novel .txt/.md documents in "
              "./manuscripts/ to reproduce that protocol.")

    meta = {}
    ids_by_domain = {}
    for name, text in texts.items():
        ids = np.array(tok.encode(text), dtype=np.int32)[:MAX_TOKENS]
        ids_by_domain[name] = ids
        np.save(os.path.join(DATA, f"{name}_ids.npy"), ids)
        meta[name] = {"chars": len(text), "tokens": int(len(ids))}
        print(f"{name}: {len(text)} chars -> {len(ids)} tokens")

    # continual stream A -> B -> A
    rel, ali = ids_by_domain["relativity"], ids_by_domain["alice"]
    seg = 15_000
    if len(rel) < 2 * seg or len(ali) < seg:
        seg = min(len(rel) // 2, len(ali))
        print(f"segment shortened to {seg}")
    aba = np.concatenate([rel[:seg], ali[:seg], rel[seg: 2 * seg]])
    np.save(os.path.join(DATA, "aba_ids.npy"), aba)
    meta["aba"] = {"tokens": int(len(aba)), "segment": int(seg)}
    print(f"aba: {len(aba)} tokens (segment {seg})")

    with open(os.path.join(DATA, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
