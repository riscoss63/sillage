"""Assemble the Kaggle kit for the GPU benches of paper 5.

Produces a flat, self-contained `kaggle_kit.zip`: the three runtime scripts
(pip `sillage` provides the package on Kaggle), the two memory states in
read-only copies, the papers corpus, and the two documents of the qwen
state's stream. The Manuscripts stream is not redistributed, so those two
documents must be supplied (--docs) unless the author's files are found
next to the repository.

    python spec/kaggle/make_kit.py --docs my_doc1.txt my_doc2.md
"""

import argparse
import os
import shutil
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
REPO = os.path.dirname(SPEC)
OUTSIDE = os.path.dirname(REPO)

STATE_FILES = ("state.npz", "cold.pkl", "log.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs=2, metavar=("DOC1", "DOC2"),
                    default=None)
    ap.add_argument("--out", default=os.path.join(os.getcwd(),
                                                  "kaggle_kit.zip"))
    a = ap.parse_args()

    docs = a.docs or [
        os.path.join(OUTSIDE, "preprint_bhd_active_inference.txt"),
        os.path.join(OUTSIDE, "PREPRINT_V2_1_BHD_ACTIVE_INFERENCE.md")]
    for d in docs:
        if not os.path.exists(d):
            raise SystemExit(f"document introuvable : {d} -- passer --docs "
                             f"(le flux Manuscripts n'est pas redistribue)")

    for src, hint in ((os.path.join(REPO, "memory_state"),
                       "sillage read doc1 doc2 --model qwen --state memory_state"),
                      (os.path.join(REPO, "papers_state", "memory"),
                       "sillage papers --with-memory")):
        if not os.path.isdir(src):
            raise SystemExit(
                f"etat absent : {src} -- les etats ne sont pas distribues "
                f"(un cold store revele ce qu'il a lu) ; "
                f"construisez-le :\n  {hint}")

    tmp = tempfile.mkdtemp()
    kit = os.path.join(tmp, "kaggle_kit")
    os.makedirs(os.path.join(kit, "states"))
    os.makedirs(os.path.join(kit, "docs"))
    for f in ("sillage_drafter.py", "torch_readout.py", "bench_gpu.py"):
        shutil.copy(os.path.join(SPEC, f), kit)
    shutil.copy(os.path.join(HERE, "README_KAGGLE.md"), kit)
    for src, name in ((os.path.join(REPO, "memory_state"), "memory_state"),
                      (os.path.join(REPO, "papers_state", "memory"),
                       "papers_state_memory")):
        dst = os.path.join(kit, "states", name)
        os.makedirs(dst)
        for f in STATE_FILES:
            shutil.copy(os.path.join(src, f), dst)
    shutil.copy(docs[0], os.path.join(kit, "docs", "manuscrit_v1.txt"))
    shutil.copy(docs[1], os.path.join(kit, "docs", "manuscrit_v2.md"))
    shutil.copy(os.path.join(REPO, "papers_state", "corpus.txt"),
                os.path.join(kit, "docs", "corpus_papers.txt"))

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(kit):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, tmp))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"kit -> {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
