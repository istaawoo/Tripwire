"""Precompute d3-flash stage-1 verdicts on a GPU (Colab) so laptops can run the cascade.

  STAGE1_BACKEND=d3 python -m monitor.d3_cache data/sample_traces.jsonl data/traces.jsonl data/traces_evasion.jsonl

Appends {trace_id, cot_len, probs, in_tokens, latency_ms} to results/stage1_d3_cache.jsonl.
Resume-safe: skips trace ids already cached. Stores probabilities, so the threshold can be tuned later.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("STAGE1_BACKEND", "d3")

from common.schemas import Trace  # noqa: E402

from . import stage1_d3  # noqa: E402


def main(paths):
    stage1_d3.CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set(stage1_d3.load_cache())
    shown = False
    for path in paths:
        if not Path(path).exists():
            print(f"missing {path}, skipping")
            continue
        traces = [Trace(**json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        for t in traces:
            if t.id in done:
                continue
            state, questions, sents = stage1_d3.build(t)
            t0 = time.perf_counter()
            try:
                raw = stage1_d3.raw_call(state, questions)
                if not shown:
                    print("RAW d3 answers (first trace):\n" + json.dumps(raw.get("answers"), indent=1, default=str)[:2500])
                    shown = True
                probs = stage1_d3.parse(raw["answers"])
            except Exception as e:
                print(f"SKIP {t.id}: {e!r:.300}")
                continue
            row = {"trace_id": t.id, "cot_len": len(t.cot), "probs": probs,
                   "in_tokens": (len(state) + len(json.dumps(questions))) // 4,
                   "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            with open(stage1_d3.CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            done.add(t.id)
            r = stage1_d3.to_result(probs, sents, 0.3)
            print(f"{t.id:<36} {t.label:<9} p={probs['p_violation']:.3f} {r.category:<20} {row['latency_ms']:.0f}ms")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/sample_traces.jsonl", "data/traces.jsonl", "data/traces_evasion.jsonl"])
