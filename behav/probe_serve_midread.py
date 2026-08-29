"""How long does a conversation wait while `sillage serve` reads a document?

The 1.6.0 release claimed "a reply in 3.3 s during an ingestion". A
real-world trial measured 113 s against a 15 s idle baseline, and found
why: `between_windows()` -- the point where the reader hands its lock
back -- was called only at 1024-token window boundaries, so a document
that fits in one window offered no yield point at all and the generation
lock was held for the whole read.

PREDICTIONS, REGISTERED BEFORE THE RUN
  P1  Yielding every 32 tokens (normal read) and after every 64-token
      block (fast read) instead of once per window brings the mid-read
      wait close to the idle baseline.
      REFUTED IF the mid-read reply takes more than 3x the idle baseline
      on a single-window document.
  P2  It costs throughput, because the lock is handed over far more often.
      REFUTED IF the read is more than 25% slower than the same read with
      the yield disabled.

    python behav/probe_serve_midread.py [--model gpt2]
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break

from sillage import Sillage                                   # noqa: E402
from sillage.serve import Service, Handler                    # noqa: E402
from http.server import ThreadingHTTPServer                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
a = ap.parse_args()

WORK = os.path.join(HERE, ".midread_state")
DOC = os.path.join(HERE, "_midread_doc.md")
import shutil                                                 # noqa: E402
shutil.rmtree(WORK, ignore_errors=True)

# one window is 1024 tokens; keep the document under it, which is the case
# the old yield point could not serve at all
open(DOC, "w", encoding="utf-8").write(
    "# Field notes\n\n"
    + ("The Zylkorb regulator on bay four holds seventeen turquoise "
       "brackets, and the Ferncastle valve is torqued to sixty-two "
       "newton-metres before the amber cipher is logged. ") * 22)

s = Sillage(model=a.model, state=WORK, quiet=True)
s.load_model()
svc = Service(s, k=3)
Handler.service = svc
httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
PORT = httpd.server_address[1]
BASE = "http://127.0.0.1:%d" % PORT
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


CHAT = {"messages": [{"role": "user", "content": "The regulator holds"}],
        "max_tokens": 12, "temperature": 0}

try:
    t0 = time.time()
    post("/v1/chat/completions", CHAT)
    idle = time.time() - t0
    print("idle chat            : %.2f s" % idle)

    task = post("/read", {"paths": [DOC], "fast": False})
    t0 = time.time()
    waits = []
    for _ in range(3):
        t1 = time.time()
        post("/v1/chat/completions", CHAT)
        waits.append(time.time() - t1)
    read_wall = None
    for _ in range(600):
        st = json.load(urllib.request.urlopen(BASE + "/tasks/" + task["task_id"]))
        if st.get("state") in ("done", "failed"):
            read_wall = time.time() - t0
            break
        time.sleep(0.5)
    print("mid-read chats       : %s s" % [round(w, 2) for w in waits])
    print("read wall            : %.1f s (%s)"
          % (read_wall or -1, st.get("state")))
    worst = max(waits)
    ratio = worst / max(1e-6, idle)
    print("\n--- verdicts against the registered predictions ---")
    print("P1 mid-read wait near idle : %s (worst %.2f s = %.1fx the "
          "%.2f s baseline)"
          % ("HELD" if ratio <= 3.0 else "REFUTED", worst, ratio, idle))
    print("P2 throughput cost         : read of %s tokens in %.1f s"
          % (st.get("done", [{}])[0].get("tokens", "?"), read_wall or -1))
    res = os.path.join(HERE, "results")
    os.makedirs(res, exist_ok=True)
    out = os.path.join(res, "serve_midread.json")
    json.dump({"model": a.model, "idle_s": round(idle, 2),
               "midread_s": [round(w, 2) for w in waits],
               "worst_ratio": round(ratio, 2),
               "read_wall_s": round(read_wall or -1, 1),
               "tokens": st.get("done", [{}])[0].get("tokens")},
              open(out, "w", encoding="utf-8"), indent=1)
    print("saved -> %s" % out)
finally:
    httpd.shutdown()
    shutil.rmtree(WORK, ignore_errors=True)
    os.path.exists(DOC) and os.remove(DOC)
