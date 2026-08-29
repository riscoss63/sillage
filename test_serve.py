"""End-to-end tests of `sillage serve`, over real HTTP.

Starts the server in a thread, talks to it with urllib (no client
library, the same way any OpenAI-compatible tool would), and checks the
four use cases the endpoint exists for:

  UC1  a chat client that was never told about Sillage gets an answer,
       and is told which sources went into the prompt
  UC2  an editor gets a raw completion
  UC3  a folder can be read in the background while the conversation
       keeps answering -- the point of releasing the lock per window
  UC4  the memory can be inspected and quoted without generating

    python test_serve.py
"""

import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sillage import Sillage                                    # noqa: E402
from sillage.serve import Handler, Service                     # noqa: E402
from http.server import ThreadingHTTPServer                    # noqa: E402

PORT = 8731
BASE = f"http://127.0.0.1:{PORT}"
STATE = "_serve_state"
DOC = "_serve_doc.md"
DOC2 = "_serve_doc2.md"
passed = []

FACTS = """# Field notes

The Zylkorb protocol requires seventeen turquoise brackets before any
maintenance window opens. Nobody outside the team knows this.

The Ilvress procedure stores the amber cipher in the west vault, and
the vault is opened only by the duty officer.
"""
FILLER = ("The committee reviewed the quarterly report on day {0} of "
          "the session, then adjourned without further comment. ")


def call(path, payload=None, method=None, timeout=180):
    url = BASE + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read()), dict(r.headers)


def check(name, cond, detail=""):
    assert cond, f"{name} FAILED {detail}"
    passed.append(f"{name} ok {detail}")


# ---------------------------------------------------------------- setup
shutil.rmtree(STATE, ignore_errors=True)
open(DOC, "w", encoding="utf-8").write(FACTS)
open(DOC2, "w", encoding="utf-8").write(
    "# Session log\n\n" + "".join(FILLER.format(i) for i in range(1, 90)))

print("building a small memory (gpt2, fast read) ...", flush=True)
s = Sillage(model="gpt2", state=STATE, quiet=True)
s.read(DOC, fast=True)

Handler.service = Service(s, context=True, k=2)
httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
httpd.quiet = True
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.3)

try:
    # ------------------------------------------------------------ UC4
    code, models, _ = call("/v1/models")
    check("S1 OpenAI model list", code == 200
          and models["data"][0]["id"] == "gpt2",
          f"({models['data'][0]['id']})")

    code, st, _ = call("/status")
    check("S2 status", code == 200 and st["tokens"] > 0
          and st["passages"] > 0,
          f"({st['tokens']} tokens, {st['passages']} passages, "
          f"v{st['version']})")

    code, got, _ = call("/ask", {"query": "amber cipher", "k": 1})
    check("S3 grounded quote (UC4)", code == 200
          and got["passages"] and "Ilvress" in got["passages"][0]["text"]
          and got["passages"][0]["source"] == DOC,
          f"(quoted {DOC} without loading anything)")

    # ------------------------------------------------------------ UC1
    code, out, hdr = call("/v1/chat/completions", {
        "model": "gpt2", "max_tokens": 12,
        "messages": [{"role": "user",
                      "content": "What does the Zylkorb protocol "
                                 "require?"}]})
    content = out["choices"][0]["message"]["content"]
    srcs = [x["source"] for x in out["sillage"]["sources"]]
    check("S4 chat completion (UC1)", code == 200 and content
          and out["object"] == "chat.completion",
          f"({len(content)} chars back, {out['sillage']['seconds']}s)")
    check("S5 sources are declared", DOC in srcs
          and hdr.get("X-Sillage-Sources", "").startswith(DOC),
          f"(header X-Sillage-Sources: {hdr.get('X-Sillage-Sources')})")
    # the shape a real client parses: no template token in the content, a
    # finish_reason that distinguishes a finished answer from a cut one,
    # and the usage object several clients require
    stops = [s for s in ("<|im_end|>", "<|endoftext|>", "</s>")
             if s in content]
    usage = out.get("usage") or {}
    check("S5b the OpenAI response shape a client can parse",
          not stops and out["choices"][0]["finish_reason"] in
          ("stop", "length")
          and usage.get("total_tokens") ==
          usage.get("prompt_tokens", 0) + usage.get("completion_tokens", -1),
          f"(finish_reason {out['choices'][0]['finish_reason']}, "
          f"usage {usage.get('total_tokens')} tokens, no template token "
          f"in the content)")

    # ------------------------------------------------------------ UC2
    code, out, _ = call("/v1/completions", {
        "prompt": "The Zylkorb protocol requires", "max_tokens": 8})
    check("S6 raw completion (UC2)", code == 200
          and out["choices"][0]["text"],
          f"({out['choices'][0]['text'][:40]!r})")

    # streaming, the shape clients expect
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"stream": True, "max_tokens": 8,
                         "messages": [{"role": "user",
                                       "content": "amber cipher?"}]}
                        ).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = r.read().decode()
    frames = [l for l in body.splitlines() if l.startswith("data: ")]
    check("S7 streaming", len(frames) >= 2
          and frames[-1] == "data: [DONE]"
          and "delta" in json.loads(frames[0][6:])["choices"][0],
          f"({len(frames)} SSE frames, terminated)")

    # ------------------------------------------------------------ UC3
    code, task, _ = call("/read", {"paths": [DOC2], "fast": True})
    check("S8 background read accepted (UC3)", code == 202
          and task["state"] == "queued", f"(task {task['task_id']})")

    t0 = time.time()
    code, out, _ = call("/v1/chat/completions", {
        "max_tokens": 6,
        "messages": [{"role": "user", "content": "amber cipher?"}]})
    waited = time.time() - t0
    check("S9 answers DURING ingestion (UC3)", code == 200
          and out["choices"][0]["message"]["content"] is not None,
          f"(replied in {waited:.1f}s while a document was being read)")

    for _ in range(600):
        code, t, _ = call(f"/tasks/{task['task_id']}")
        if t["state"] in ("done", "failed"):
            break
        time.sleep(0.5)
    check("S10 ingestion completes", t["state"] == "done"
          and t["done"] and t["done"][0]["tokens"] > 100,
          f"({t['done'][0]['tokens']} tokens ingested, "
          f"state {t['state']})")

    code, st2, _ = call("/status")
    check("S11 memory grew and was saved", st2["tokens"] > st["tokens"]
          and os.path.exists(os.path.join(STATE, "state.npz")),
          f"({st['tokens']} -> {st2['tokens']} tokens on disk)")

    # ------------------------------------------------------- erreurs
    try:
        call("/read", {"paths": ["does_not_exist.md"]})
        ok404 = False
    except urllib.error.HTTPError as e:
        ok404 = e.code == 400 and b"no such file" in e.read()
    check("S12 refuses a missing file", ok404,
          "(400 with the name, not a stack trace)")

    try:
        call("/v1/chat/completions", {"messages": []})
        okempty = False
    except urllib.error.HTTPError as e:
        okempty = e.code == 400
    check("S13 refuses an empty request", okempty, "(400)")

    # ------------------------------------------------------ auth mode
    Handler.service.token = "s3cret"
    try:
        call("/status")
        okauth = False
    except urllib.error.HTTPError as e:
        okauth = e.code == 401
    req = urllib.request.Request(
        BASE + "/status", headers={"Authorization": "Bearer s3cret"})
    with urllib.request.urlopen(req, timeout=30) as r:
        okauth = okauth and r.status == 200
    Handler.service.token = None
    check("S14 bearer token", okauth,
          "(401 without, 200 with)")

finally:
    httpd.shutdown()
    httpd.server_close()
    shutil.rmtree(STATE, ignore_errors=True)
    for f in (DOC, DOC2):
        if os.path.exists(f):
            os.remove(f)

print("\n".join(passed))
print(f"\nALL {len(passed)} SERVE TESTS PASSED")
