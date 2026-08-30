"""The composition, not the channel: does `serve` answer a rephrased question?

`probe_chattemplate` isolated the READOUT under a chat template and found
it worse there, not better. But that is not what `serve` does. `serve`
also injects the passages `ask` retrieves -- and that channel IS robust
to rephrasing, because it is TF-IDF over the querent's own words rather
than a four-token key (entry@1 is 12/12). Paper 7 measured exactly this
split: 5% for the memory answering alone against 25% for the same
evidence placed in the window.

So this asks the real endpoint, over real sockets, with `--no-context`
as the ablation that separates the two channels.

Registered BEFORE the run:

  S1  With context injection, rephrased recall is at least 6/8 at 0.6B
      -- the passage is in the window and the model only has to read it.
      FALSIFIED below 5/8.
  S2  The gain is the RETRIEVAL channel: `--no-context` scores strictly
      lower than the same model with context.
      FALSIFIED if they tie or invert -- then the readout was doing the
      work and paper 7's split does not hold here.
  S3  Unanswerable questions stay refused or unsupported: at most 2 of 8
      produce a confident false fact WITH a source named. This is the
      dangerous combination, so it is the number that decides whether
      `serve` is usable by someone who is not you.
  S4  1.7B with context >= 0.6B with context. A bigger reader should
      help HERE if it helps anywhere, because the answer is in front of
      it and the task is reading rather than recall.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sillage.runtime import Sillage                      # noqa: E402
from probe_abstain_gen import DOC, REWORDED              # noqa: E402
from probe_chattemplate import UNANSWERABLE_Q            # noqa: E402

PORT = 8137
MAXTOK = 48


def fold(s):
    """Accent-blind, punctuation-blind comparison: `quatrieme` must match
    `quatrieme`, and a fact answered as `38` must match `trente-huit`."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s)


#: the same fact, written the way a model may choose to write it
ALIASES = {"trente-huit": ["trente huit", "38"],
           "quatorze": ["14"], "huit": ["8"], "trois": ["3"],
           "quatrieme": ["4e", "4 eme", "quatrieme"],
           "11 avril": ["11 avril"], "2,4": ["2 4"], "Ovide": ["ovide"]}


def says(want, txt):
    f = fold(txt)
    return any(fold(v) in f for v in [want] + ALIASES.get(want, []))


def post(port, question):
    body = json.dumps({"messages": [{"role": "user", "content": question}],
                       "max_tokens": MAXTOK, "temperature": 0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def wait(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server died: {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                   timeout=5).read()
            return True
        except Exception:
            time.sleep(3)
    raise SystemExit("server never came up")


def arm(state, label, port, extra):
    cmd = [sys.executable, "-m", "sillage", "serve", "--state", state,
           "--port", str(port), "--model", "qwen", "--quiet"] + extra
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    rows = []
    try:
        wait(port, proc)
        for q, want in ([(q, w) for q, w in REWORDED]
                        + [(q, None) for q in UNANSWERABLE_Q]):
            d = post(port, q)
            txt = d["choices"][0]["message"]["content"]
            src = [s["source"] for s in
                   (d.get("sillage") or {}).get("sources", [])]
            rows.append({"q": q[:44], "want": want, "got": txt.strip()[:90],
                         "correct": bool(want and says(want, txt)),
                         "sources": src})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    ok = sum(r["correct"] for r in rows if r["want"])
    # a false fact that arrives WITH a source named is the dangerous case
    sourced_una = sum(1 for r in rows if not r["want"] and r["sources"])
    print(f"  {label:<34} rephrased {ok}/8   unanswerable-with-a-source "
          f"{sourced_una}/8", flush=True)
    for r in rows:
        tag = ("OK " if r["correct"] else
               ("?? " if r["want"] else "   "))
        print(f"      {tag}{(r['want'] or 'NO ANSWER'):<12} "
              f"{r['got'][:64]!r}", flush=True)
    return {"rows": rows, "correct": ok, "sourced_unanswerable": sourced_una}


def main():
    state = tempfile.mkdtemp(prefix="serverep_")
    res = {}
    try:
        print("reading the report ...", flush=True)
        s = Sillage(model="qwen", state=state, quiet=True)
        text = Sillage.reflow(DOC)
        for _ in range(2):
            s.read_text(text)
        s.index.add(text, "rucher.md")
        s.save()
        print(f"  {len(s.mem.cold)} grams, {len(s.index.passages)} passages",
              flush=True)
        del s

        res["0.6B +context"] = arm(state, "0.6B, passages injected",
                                   PORT, [])
        res["0.6B -context"] = arm(state, "0.6B, --no-context (readout only)",
                                   PORT + 1, ["--no-context"])
        # `serve` does NOT accept --target: that flag is declared in the
        # `gen` argument group, which the serve subparser does not inherit.
        # So the endpoint cannot be pointed at a bigger reader at all --
        # a real gap, recorded here rather than worked around.
        res["1.7B +context"] = {"rows": [], "correct": None,
                                "sourced_unanswerable": None,
                                "note": "serve does not accept --target"}
    finally:
        shutil.rmtree(state, ignore_errors=True)

    a, b, c = (res["0.6B +context"], res["0.6B -context"],
               res["1.7B +context"])
    _ = c
    v = {"S1": {"with_context_06B": a["correct"], "holds": a["correct"] >= 5},
         "S2": {"with": a["correct"], "without": b["correct"],
                "holds": a["correct"] > b["correct"]},
         "S3": {"unanswerable_with_a_source": a["sourced_unanswerable"],
                "holds": a["sourced_unanswerable"] <= 2},
         "S4": {"note": "not measurable: serve has no --target",
                "06B": a["correct"], "holds": None}}
    print("\n" + json.dumps(v, indent=1, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "serve_rephrase.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump({"arms": res, "verdict": v}, fh, indent=1,
                  ensure_ascii=False)
    print(f"written {out}")


if __name__ == "__main__":
    main()
