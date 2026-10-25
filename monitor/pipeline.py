"""Cascade monitor.  python -m monitor.pipeline data/sample_traces.jsonl [--baseline]

analyze(trace)             stage 1 on every trace, stage 2 only on flags.
analyze_explain_all(trace) baseline: stage 2 on every trace with the full CoT, no stage 1.
                           Its `stage1` field is a placeholder carrying stage 2's verdict
                           (flag=confirmed, category from stage 2, confidence 1.0/0.0) so
                           per-category metrics still work. Tokens are all in stage2_*.
"""
import json
import sys
import time

from common.schemas import MonitorResult, Stage1Result, TokenUsage, Trace, to_dict

from . import stage1, stage2


def analyze(trace: Trace) -> MonitorResult:
    t0 = time.perf_counter()
    tok = TokenUsage()
    s1, tok.stage1_in, tok.stage1_out = stage1.classify(trace)
    s2 = None
    if s1.flag:
        s2, tok.stage2_in, tok.stage2_out = stage2.explain(trace, s1)
    return MonitorResult(
        trace_id=trace.id, stage1=s1, stage2=s2,
        final_flag=bool(s1.flag and (s2 is None or s2.confirmed)),
        tokens=tok, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def analyze_explain_all(trace: Trace) -> MonitorResult:
    t0 = time.perf_counter()
    s2, cat, tin, tout = stage2.explain_full(trace)
    placeholder = Stage1Result(flag=s2.confirmed, category=cat if s2.confirmed else "none",
                               confidence=1.0 if s2.confirmed else 0.0, suspicious_span="")
    return MonitorResult(
        trace_id=trace.id, stage1=placeholder, stage2=s2, final_flag=bool(s2.confirmed),
        tokens=TokenUsage(stage2_in=tin, stage2_out=tout),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


def load_traces(path):
    with open(path, encoding="utf-8") as f:
        return [Trace(**json.loads(line)) for line in f if line.strip()]


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    fn = analyze_explain_all if "--baseline" in argv else analyze
    path = [a for a in argv if not a.startswith("--")][0]
    traces = load_traces(path)
    hits = tok_total = 0
    for t in traces:
        try:
            r = fn(t)
        except Exception as e:
            print(f"SKIP {t.id}: {e!r:.200}")
            continue
        n = r.tokens.stage1_in + r.tokens.stage1_out + r.tokens.stage2_in + r.tokens.stage2_out
        tok_total += n
        truth = t.label == "malicious"
        hits += r.final_flag == truth
        mark = "OK " if r.final_flag == truth else "MISS" if truth else "FP "
        if r.stage2 is None:
            s2 = ""
        elif fn is analyze_explain_all:
            s2 = " s2=flag" if r.stage2.confirmed else " s2=clear"
        else:
            s2 = " s2=confirm" if r.stage2.confirmed else " s2=OVERTURN"
        print(f"{mark} {t.id:<34} {t.label:<9} {t.difficulty:<13} s1={r.stage1.confidence:.2f} {r.stage1.category:<20}{s2:<13} "
              f"final={r.final_flag!s:<5} tok={n:<5} {r.latency_ms:.0f}ms")
        if "--json" in argv:
            print(json.dumps(to_dict(r)))
    print(f"\n{hits}/{len(traces)} correct, {tok_total} tokens total ({'baseline' if fn is analyze_explain_all else 'cascade'})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
