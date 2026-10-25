"""Cascade monitor: stage 1 on every trace, stage 2 only on flagged ones.

    python -m monitor.pipeline data/sample_traces.jsonl
"""
import json
import sys
import time

from common.schemas import MonitorResult, Stage1Result, TokenUsage, Trace, to_dict
from monitor.stage1 import classify
from monitor.stage2 import explain, explain_full

# Team decision: stage 1 (d3-flash, 9B) is stronger than stage 2 (MiniCPM5-2B), so
# stage 2 only explains and cannot clear a flag. stage2.confirmed still records
# whether MiniCPM agreed ("would have overturned" = confirmed is False).
STAGE2_CAN_OVERTURN = False


def analyze(trace: Trace) -> MonitorResult:
    t0 = time.perf_counter()
    tokens = TokenUsage()
    s1, tokens.stage1_in, tokens.stage1_out = classify(trace)
    s2 = None
    if s1.flag:
        s2, tokens.stage2_in, tokens.stage2_out = explain(trace, s1)
    if STAGE2_CAN_OVERTURN:
        final = s1.flag and (s2 is None or s2.confirmed)
    else:
        final = s1.flag
    return MonitorResult(trace.id, s1, s2, final, tokens, (time.perf_counter() - t0) * 1000)


def analyze_explain_all(trace: Trace) -> MonitorResult:
    """Baseline: always run stage 2 on the full CoT, no stage 1."""
    t0 = time.perf_counter()
    tokens = TokenUsage()
    s2, category, tokens.stage2_in, tokens.stage2_out = explain_full(trace)
    # stage1 slot records the baseline's own verdict so the schema stays uniform
    s1 = Stage1Result(s2.confirmed, category, 1.0 if s2.confirmed else 0.0, "")
    return MonitorResult(trace.id, s1, s2, s2.confirmed, tokens, (time.perf_counter() - t0) * 1000)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_traces.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = Trace(**json.loads(line))
            r = analyze(t)
            mark = "FLAG" if r.final_flag else "ok  "
            print(f"{mark} {t.id:45s} truth={t.label:9s} s1={r.stage1.category}:{r.stage1.confidence:.2f} "
                  f"span={r.stage1.suspicious_span[:50]!r}")
            print("     " + json.dumps(to_dict(r))[:300])
