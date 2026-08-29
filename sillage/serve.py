"""`sillage serve`: the memory behind an OpenAI-compatible endpoint.

Point any client that speaks OpenAI at http://127.0.0.1:8000/v1 and it
gains the memory of what you have read -- no integration, no plugin, and
no dependency beyond the three the tool already has (this is the Python
standard library's own HTTP server; a local single-user service does not
need a web framework).

    sillage serve                       # 127.0.0.1:8000, ./.sillage
    sillage serve --port 9000 --state ~/notes-memory

What the memory does to an answer, and why. Two mechanisms, both on by
default, because paper 7 measured what each is worth:

  * CONTEXT -- the passages your question matches are prepended to the
    prompt, with their source. On LongMemEval the memory alone answered
    5% of questions while the same evidence in the window answered 25%:
    formulation happens in the window, so the window is where the
    evidence goes. Every answer says which sources were used, in the
    `sillage` field of the response and in the X-Sillage-Sources header:
    a proxy that silently rewrites your prompt is not auditable.
  * READOUT -- the memory also mixes into the model's own next-token
    distribution, which is what makes it recall wording it has read.
    Paper 7 also measured that this never hurts when the evidence is
    already in the window (identical answers on 40 of 40 questions), so
    running both is free.

Endpoints:
    GET  /v1/models                 the served model (OpenAI shape)
    POST /v1/chat/completions       chat, with `stream` supported
    POST /v1/completions            raw completion, for editors
    POST /read      {"paths": [...], "fast": true}  -> a task id
    GET  /tasks/<id>                progress of one ingestion
    GET  /status                    what the memory knows, tier by tier
    POST /ask       {"query": "...", "k": 3}        grounded passages

Concurrency, and the one thing that matters: a Sillage state is not
thread-safe -- reading mutates the matrices in place. Generation and
ingestion therefore share one lock, and the ingestion RELEASES it
between windows (see `between_windows` in runtime.read_text). So a
conversation stays answerable while a folder is being read; the longest
it ever waits is one window.

Binding: 127.0.0.1 by default. `--host 0.0.0.0` exposes a memory that
contains the text you fed it -- the server says so out loud, and
`--token` adds a bearer check if you do it anyway.
"""

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__

MAX_BODY = 4 * 1024 * 1024


class Service:
    """The shared state: one assistant, one lock, a task table."""

    def __init__(self, assistant, context=True, k=3, token=None):
        self.s = assistant
        self.context = context          # inject retrieved passages
        self.k = k
        self.token = token
        self.lock = threading.Lock()
        self.waiting = 0                # generations queued behind a read
        self.tasks = {}
        self.tasks_lock = threading.Lock()

    # ---- the yield point an ingestion calls between windows ----------
    def _yield_lock(self):
        if self.waiting:
            self.lock.release()
            time.sleep(0.01)            # let a waiter actually take it
            self.lock.acquire()

    def generate(self, prompt, n, temp, seed):
        """Take the lock only for the generation itself."""
        self.waiting += 1
        try:
            with self.lock:
                return self.s.complete(prompt, n=n, temp=temp, seed=seed)
        finally:
            self.waiting -= 1

    def ask(self, query, k=None):
        self.waiting += 1
        try:
            with self.lock:
                hits = self.s.ask(query, k=k or self.k)
        finally:
            self.waiting -= 1
        return [{"score": round(float(sc), 4),
                 "source": p["source"],
                 "section": p.get("section"),
                 "text": p["text"]} for sc, p in hits]

    def status(self):
        with self.lock:
            st = self.s.status()
        st["version"] = __version__
        st["context_injection"] = self.context
        return st

    # ---- ingestion, in the background --------------------------------
    def start_read(self, paths, fast=True):
        task = {"id": uuid.uuid4().hex[:12], "state": "queued",
                "paths": list(paths), "fast": bool(fast),
                "done": [], "started": time.time(), "error": None}
        with self.tasks_lock:
            self.tasks[task["id"]] = task
        threading.Thread(target=self._run_read, args=(task,),
                         daemon=True).start()
        return task["id"]

    def _run_read(self, task):
        task["state"] = "reading"
        try:
            for path in task["paths"]:
                with self.lock:
                    rec = self.s.read(
                        path, save=False, fast=task["fast"],
                        between_windows=self._yield_lock)[0]
                    self.s.save()
                task["done"].append(
                    {"file": rec["file"], "tokens": rec["tokens"],
                     "minutes": rec.get("minutes"),
                     "ppl_with_memory": rec.get("ppl_with_memory")})
            task["state"] = "done"
        except Exception as exc:                       # noqa: BLE001
            task["state"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
        task["finished"] = time.time()

    def task(self, tid):
        with self.tasks_lock:
            return self.tasks.get(tid)

    # ---- prompt assembly --------------------------------------------
    def build_prompt(self, messages_or_text, is_chat):
        """Return (prompt, sources). Paper 7: put the evidence in the
        window, and say that you did.

        An instruction-tuned model answers badly when handed raw
        "role: content" lines, so its own chat template is used when it
        has one -- that is the difference between a demo and something
        a client can actually talk to.
        """
        question = ""
        if is_chat:
            msgs = [dict(m) for m in messages_or_text]
            for m in reversed(msgs):
                if m.get("role") == "user":
                    question = str(m.get("content", ""))
                    break
        else:
            question = str(messages_or_text)

        # retrieve once, use twice (sources reported, passages injected)
        sources, passages = [], ""
        if self.context and question.strip():
            hits = self.ask(question, self.k)
            sources = [{"source": h["source"], "score": h["score"]}
                       for h in hits]
            passages = "\n\n".join(f"[{h['source']}] {h['text']}"
                                   for h in hits)

        if not is_chat:
            body = str(messages_or_text)
            if passages:
                body = (f"Notes from what I have read:\n\n{passages}"
                        f"\n\n---\n\n{body}")
            return body, sources

        if passages:
            note = {"role": "system",
                    "content": "Notes from what I have read; use them "
                               "if they are relevant.\n\n" + passages}
            msgs = [note] + msgs
        tok, _ = self.s.load_model()
        template = getattr(tok, "chat_template", None)
        if template:
            try:
                return tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False), sources
            except TypeError:          # older templates: no thinking arg
                return tok.apply_chat_template(
                    msgs, tokenize=False,
                    add_generation_prompt=True), sources
            except Exception:          # a broken template must not 500
                pass
        body = "\n".join(f"{m.get('role', 'user')}: {m.get('content')}"
                         for m in msgs)
        return body + "\nassistant:", sources


def _json(handler, code, payload, extra=None):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    for k, v in (extra or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = f"sillage/{__version__}"
    service = None                     # set by serve()

    def log_message(self, fmt, *args):        # one tidy line per call
        if not self.server.quiet:
            print(f"  {self.command} {self.path} -> {args[1]}",
                  flush=True)

    # ---- helpers -----------------------------------------------------
    def _authorised(self):
        want = self.service.token
        if not want:
            return True
        got = self.headers.get("Authorization", "")
        return got.strip() == f"Bearer {want}"

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return None
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return None

    # ---- routes ------------------------------------------------------
    def do_GET(self):
        if not self._authorised():
            return _json(self, 401, {"error": "bad or missing token"})
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            model = self.service.s.mem.which
            return _json(self, 200, {"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "sillage"}]})
        if path == "/status":
            return _json(self, 200, self.service.status())
        if path.startswith("/tasks/"):
            task = self.service.task(path.rsplit("/", 1)[-1])
            if task is None:
                return _json(self, 404, {"error": "no such task"})
            return _json(self, 200, task)
        if path in ("", "/"):
            return _json(self, 200, {
                "service": "sillage", "version": __version__,
                "endpoints": ["/v1/models", "/v1/chat/completions",
                              "/v1/completions", "/read", "/tasks/<id>",
                              "/status", "/ask"]})
        return _json(self, 404, {"error": "unknown endpoint"})

    def do_POST(self):
        if not self._authorised():
            return _json(self, 401, {"error": "bad or missing token"})
        path = self.path.split("?")[0].rstrip("/")
        payload = self._body()
        if payload is None:
            return _json(self, 400, {"error": "body must be JSON and "
                                              "under 4 MB"})
        if path == "/v1/chat/completions":
            return self._completions(payload, chat=True)
        if path == "/v1/completions":
            return self._completions(payload, chat=False)
        if path == "/ask":
            q = str(payload.get("query", "")).strip()
            if not q:
                return _json(self, 400, {"error": "query is required"})
            hits = self.service.ask(q, int(payload.get("k", 0)) or None)
            return _json(self, 200, {"query": q, "passages": hits})
        if path == "/read":
            paths = payload.get("paths") or []
            if isinstance(paths, str):
                paths = [paths]
            missing = [p for p in paths if not os.path.exists(p)]
            if not paths or missing:
                return _json(self, 400, {
                    "error": "no such file: " + ", ".join(missing)
                             if missing else "paths is required"})
            tid = self.service.start_read(
                paths, fast=bool(payload.get("fast", True)))
            return _json(self, 202, {"task_id": tid, "state": "queued",
                                     "poll": f"/tasks/{tid}"})
        return _json(self, 404, {"error": "unknown endpoint"})

    # ---- generation --------------------------------------------------
    def _completions(self, payload, chat):
        src = (payload.get("messages") if chat
               else payload.get("prompt"))
        if not src:
            return _json(self, 400, {
                "error": "messages are required" if chat
                         else "prompt is required"})
        n = int(payload.get("max_tokens") or 64)
        temp = float(payload.get("temperature") or 0.0)
        seed = int(payload.get("seed") or 0)
        prompt, sources = self.service.build_prompt(src, chat)
        t0 = time.time()
        text = self.service.generate(prompt, n, temp, seed)
        took = time.time() - t0
        hdr = {"X-Sillage-Sources":
               ", ".join(s["source"] for s in sources) or "none"}
        made = int(time.time())
        model = self.service.s.mem.which
        if payload.get("stream"):
            return self._stream(text, model, made, hdr, chat)
        body = {
            "id": "cmpl-" + uuid.uuid4().hex[:16],
            "object": "chat.completion" if chat else "text_completion",
            "created": made, "model": model,
            "choices": [
                {"index": 0, "finish_reason": "length",
                 **({"message": {"role": "assistant", "content": text}}
                    if chat else {"text": text})}],
            # what the memory did, in the open
            "sillage": {"sources": sources, "seconds": round(took, 2),
                        "context_injection": self.service.context},
        }
        return _json(self, 200, body, hdr)

    def _stream(self, text, model, made, hdr, chat):
        """Server-sent events, the shape OpenAI clients expect."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        for k, v in hdr.items():
            self.send_header(k, v)
        self.end_headers()
        cid = "cmpl-" + uuid.uuid4().hex[:16]
        obj = "chat.completion.chunk" if chat else "text_completion"

        def send(delta, finish=None):
            choice = {"index": 0, "finish_reason": finish}
            choice.update({"delta": {"content": delta}} if chat
                          else {"text": delta})
            frame = {"id": cid, "object": obj, "created": made,
                     "model": model, "choices": [choice]}
            self.wfile.write(f"data: {json.dumps(frame)}\n\n"
                             .encode("utf-8"))
            self.wfile.flush()

        # the text is already generated: stream it word by word so a
        # client's rendering stays lively without faking token timing
        parts = text.split(" ")
        for i, part in enumerate(parts):
            send(part if i == 0 else " " + part)
        send("", finish="length")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(assistant, host="127.0.0.1", port=8000, context=True, k=3,
          token=None, quiet=False):
    """Run until interrupted. Returns nothing; prints how to use it."""
    Handler.service = Service(assistant, context=context, k=k,
                              token=token)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.quiet = quiet
    st = Handler.service.status()
    print(f"sillage {__version__} serving {st['model']} on "
          f"http://{host}:{port}/v1")
    print(f"  memory: {st['tokens']} tokens read, "
          f"{st['passages']} passages indexed, "
          f"{st['cold_grams']} cold grams")
    inject = (f"on (top-{k} passages, sources reported)" if context
              else "off")
    print(f"  context injection: {inject}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING: bound to {host}. This memory contains the "
              f"text you fed it; anyone who can reach this port can "
              f"read it back."
              + ("" if token else " No --token is set."))
    print("  ctrl-c to stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping; saving the state ...", flush=True)
        with Handler.service.lock:
            assistant.save()
        httpd.server_close()
