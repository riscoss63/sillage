"""What this machine can actually do: throughput, footprint, portability.

Written to answer one question the project cannot answer yet -- whether
Sillage runs, and runs fast enough, on the class of hardware a physical
product would use. It is deliberately platform-neutral so the SAME
numbers can be taken on a Windows desktop, an ARM64 CI runner, a
Raspberry Pi or a cloud instance, and compared without an asterisk.

Everything is synthetic and deterministic: no corpus is downloaded, no
result depends on what happens to be on disk. GPT-2 is the model because
it is small enough to fetch in a CI job and because most of the papers'
numbers were taken with it.

Reported, in this order of importance for a device:

  read_fast   tokens/s of paper 7's blocked ingestion -- the number that
              decides whether an appliance can absorb a folder overnight
  read_normal tokens/s of the priced read (perplexity reported)
  generate    tokens/s of decoding with the memory
  rss_peak    high-water memory, where the OS will tell us
  wheels      whether the stack installed at all on this architecture

Run:  python behav/bench_platform.py [--tokens 3000] [--model gpt2]
"""
import argparse
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sillage import __version__                       # noqa: E402
from sillage.runtime import Sillage                    # noqa: E402
from sillage.ingest import ingest_text                 # noqa: E402


def rss_peak_mb():
    """High-water RSS, where the platform exposes one."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes
        return round(v / (1024 if sys.platform != "darwin" else 1024 ** 2), 1)
    except Exception:
        pass
    try:                                    # Windows, without a dependency
        import ctypes
        import ctypes.wintypes as wt

        class C(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = C()
        c.cb = ctypes.sizeof(C)
        k32 = ctypes.windll.kernel32
        # a HANDLE is 64-bit here and ctypes defaults a return type to
        # c_int, so the pseudo-handle came back truncated and the call
        # failed silently -- returning a peak of 0.0 MB
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        # modern Windows exports this from kernel32 as K32...; psapi.dll
        # is the older name and is not always the one that answers
        fn = getattr(k32, "K32GetProcessMemoryInfo",
                     None) or ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(C), wt.DWORD]
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return round(c.PeakWorkingSetSize / 1024 ** 2, 1)
    except Exception:
        return None


def doc(n_sentences):
    """Deterministic prose: no download, identical on every machine.

    This is a THROUGHPUT fixture, not a quality one. The grammar is
    small on purpose -- every machine must see byte-identical input --
    which makes the text trivially predictable, so the perplexity this
    run reports (around 1.15) says nothing about the memory and is
    recorded only to show the read really happened. Compute per token is
    what transfers between machines, and that is unaffected.
    """
    subj = ["the committee", "the board", "the council", "the panel",
            "the office", "the delegation", "the working group"]
    verb = ["reviewed", "discussed", "examined", "approved", "deferred",
            "amended", "recorded"]
    obj = ["the quarterly report", "the budget allocation",
           "the hiring plan", "the safety audit", "the vendor contract",
           "the maintenance schedule", "the training programme"]
    out = []
    for k in range(n_sentences):
        out.append(f"{subj[(k * 3) % 7].capitalize()} {verb[(k * 5) % 7]} "
                   f"{obj[(k * 11) % 7]} on day {k % 28 + 1} of "
                   f"session {k // 28 + 1}.")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy
    import torch
    import transformers

    res = {"sillage": __version__,
           "platform": {
               "system": platform.system(),
               "machine": platform.machine(),
               "processor": platform.processor() or "unknown",
               "python": platform.python_version(),
               "cpu_count": os.cpu_count(),
               "torch": torch.__version__,
               "numpy": numpy.__version__,
               "transformers": transformers.__version__,
               "threads": torch.get_num_threads()},
           "model": a.model}
    print(json.dumps(res["platform"], indent=1), flush=True)

    text = doc(max(20, a.tokens // 12))
    tmp = tempfile.mkdtemp(prefix="bench_")
    try:
        s = Sillage(model=a.model, state=tmp, quiet=True)
        t0 = time.time()
        s.load_model()
        res["model_load_s"] = round(time.time() - t0, 1)
        print(f"model loaded in {res['model_load_s']} s", flush=True)

        t0 = time.time()
        rec = ingest_text(s, text, "bench", quiet=True)
        dt = time.time() - t0
        res["read_fast"] = {"tokens": rec["tokens"], "seconds": round(dt, 2),
                            "tok_per_s": round(rec["tokens"] / dt, 1)}
        print(f"read --fast : {res['read_fast']['tok_per_s']} tok/s "
              f"({rec['tokens']} tokens in {dt:.1f} s)", flush=True)

        t0 = time.time()
        rec2 = s.read_text(text, "bench2")
        dt = time.time() - t0
        res["read_normal"] = {"tokens": rec2["tokens"],
                              "seconds": round(dt, 2),
                              "tok_per_s": round(rec2["tokens"] / dt, 1),
                              "ppl_frozen": rec2.get("ppl_frozen"),
                              "ppl_with_memory": rec2.get("ppl_with_memory"),
                              "ppl_note": "fixture is trivially "
                                          "predictable; not a quality "
                                          "measurement"}
        print(f"read normal : {res['read_normal']['tok_per_s']} tok/s "
              f"(ppl {rec2.get('ppl_frozen')} -> "
              f"{rec2.get('ppl_with_memory')})", flush=True)

        t0 = time.time()
        out = s.complete("The committee reviewed", n=a.gen)
        dt = time.time() - t0
        res["generate"] = {"tokens": a.gen, "seconds": round(dt, 2),
                           "tok_per_s": round(a.gen / dt, 2),
                           "sample": out.strip()[:60]}
        print(f"generate    : {res['generate']['tok_per_s']} tok/s",
              flush=True)

        s.save()
        res["state_bytes"] = sum(
            os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp))
        res["cold_grams"] = len(s.mem.cold)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res["rss_peak_mb"] = rss_peak_mb()
    print(f"state {res['state_bytes'] / 1e6:.1f} MB on disk, "
          f"{res['cold_grams']} grams | peak RSS "
          f"{res['rss_peak_mb']} MB", flush=True)

    out_path = a.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"bench_{platform.system().lower()}_{platform.machine().lower()}"
        f".json")
    # `--out bench.json` has no directory part, and os.makedirs("") raises
    # -- which failed the CI job AFTER the whole benchmark had run
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
