"""LongMemEval-S, arm E: the extractive voice alone, judge-free.

Every session of a question's haystack is indexed as one document (the
lexical index behind `sillage ask` -- no model, no GPU), the question
is the query. Deterministic metrics, all 500 questions:

  evidence@k     a top-k passage comes from an answer session
  answer_top3    the gold answer string (normalized) appears in the
                 top-3 passage text (strict secondary)
  coverage       multi-session only: all vs some evidence sessions hit
  _abs subset    30 questions with no evidence -- reported apart

Predictions registered in NOTES_AXE3.md before this run.

    python arm_e_extractive.py
"""

import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(HERE), HERE):
    if os.path.isdir(os.path.join(_p, "sillage")):
        sys.path.insert(0, _p)
        break
sys.path.insert(0, HERE)

from sillage.index import Index                                # noqa: E402

DATA = os.path.join(HERE, "data", "longmemeval_s")
K_LIST = (1, 3, 5)


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def session_text(turns):
    return "\n\n".join(f"{t['role']}: {t['content']}" for t in turns)


def main():
    if not os.path.exists(DATA):
        raise SystemExit(
            "LongMemEval-S not found -- download it from its "
            "authors (never redistributed here):\n"
            "  hf download xiaowu0162/longmemeval "
            "longmemeval_s --repo-type dataset "
            "--local-dir longmemeval/data")
    d = json.load(open(DATA, encoding="utf-8"))
    print(f"{len(d)} questions", flush=True)
    rows = []
    t0 = time.time()
    for qi, q in enumerate(d):
        ix = Index(None)
        for sid, turns in zip(q["haystack_session_ids"],
                              q["haystack_sessions"]):
            ix.add(session_text(turns), str(sid))
        hits = ix.search(q["question"], k=max(K_LIST))
        srcs = [h[1]["source"] for h in hits]
        ans_ids = {str(a) for a in q["answer_session_ids"]}
        row = {"id": q["question_id"], "type": q["question_type"],
               "abs": str(q["question_id"]).endswith("_abs"),
               "n_evidence": len(ans_ids)}
        for k in K_LIST:
            row[f"evidence@{k}"] = any(s in ans_ids for s in srcs[:k])
        top3 = norm(" ".join(h[1]["text"] for h in hits[:3]))
        row["answer_top3"] = norm(q["answer"]) in top3
        row["coverage"] = (len(ans_ids & set(srcs[:5]))
                           / max(1, len(ans_ids)))
        rows.append(row)
        if (qi + 1) % 100 == 0:
            rate = (qi + 1) / (time.time() - t0)
            print(f"  {qi+1}/{len(d)} ({rate:.1f} q/s)", flush=True)

    core = [r for r in rows if not r["abs"]]
    absq = [r for r in rows if r["abs"]]

    def agg(rs, key):
        return sum(r[key] for r in rs) / max(1, len(rs))

    R = {"n": len(rows), "n_core": len(core), "n_abs": len(absq),
         "overall": {f"evidence@{k}": agg(core, f"evidence@{k}")
                     for k in K_LIST}}
    R["overall"]["answer_top3"] = agg(core, "answer_top3")
    R["by_type"] = {}
    for t in sorted({r["type"] for r in core}):
        rs = [r for r in core if r["type"] == t]
        R["by_type"][t] = {"n": len(rs),
                           "evidence@3": agg(rs, "evidence@3"),
                           "answer_top3": agg(rs, "answer_top3")}
    ms = [r for r in core if r["type"] == "multi-session"]
    R["multi_session_coverage"] = {
        "full@5": sum(r["coverage"] == 1.0 for r in ms) / max(1, len(ms)),
        "partial@5": sum(0 < r["coverage"] < 1.0 for r in ms)
        / max(1, len(ms)),
        "none@5": sum(r["coverage"] == 0.0 for r in ms) / max(1, len(ms))}
    R["abs_subset"] = {"n": len(absq),
                       "answer_top3": agg(absq, "answer_top3")}

    print("\n== global (hors _abs) ==")
    for k in K_LIST:
        print(f"  evidence@{k} : {R['overall'][f'evidence@{k}']:.1%}")
    print(f"  answer_top3 : {R['overall']['answer_top3']:.1%}")
    print("== par type ==")
    for t, v in R["by_type"].items():
        print(f"  {t:28s} n={v['n']:3d}  evidence@3 {v['evidence@3']:.1%}"
              f"  answer_top3 {v['answer_top3']:.1%}")
    print(f"== multi-session coverage@5 == {R['multi_session_coverage']}")

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "lme_arm_e.json")
    json.dump({"metrics": R, "rows": rows}, open(out, "w"), indent=1)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
