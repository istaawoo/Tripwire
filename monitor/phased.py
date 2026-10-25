"""Two-phase run so only one model needs to be on the GPU at a time.

    python -m monitor.phased stage1 [files...] [--minutes N]   # d3 server only
    python -m monitor.phased stage2 [files...] [--minutes N]   # MiniCPM server only

Phase 1 caches stage-1 verdicts to results/stage1_d3_cache.jsonl.
Phase 2 builds results/monitor_results.jsonl (cascade, from the cache) and
results/baseline_results.jsonl (explain-all), alternating per trace so a cut-off
run still leaves a fair, same-traces comparison. Both phases are resume-safe and
write after every trace, so whatever exists at the deadline is usable.
"""
import argparse
import json
import time
from dataclasses import dataclass

from tqdm import tqdm

from common.schemas import MonitorResult, Stage1Result, TokenUsage, Trace, to_dict
from eval.loaders import BASELINE_RESULTS, MONITOR_RESULTS, RESULTS, load_traces, read_jsonl
from monitor.pipeline import STAGE2_CAN_OVERTURN, analyze_explain_all
from monitor.stage1 import classify
from monitor.stage2 import explain

STAGE1_CACHE = RESULTS / "stage1_d3_cache.jsonl"


@dataclass
class Stage1Cache:
    trace_id: str
    stage1: Stage1Result
    stage1_in: int
    stage1_out: int
    latency_ms: float


def cascade_from_cache(trace: Trace, c: dict) -> MonitorResult:
    t0 = time.perf_counter()
    s1 = Stage1Result(**c["stage1"])
    tokens = TokenUsage(stage1_in=c["stage1_in"], stage1_out=c["stage1_out"])
    s2 = None
    if s1.flag:
        s2, tokens.stage2_in, tokens.stage2_out = explain(trace, s1)
    final = s1.flag and (s2 is None or s2.confirmed) if STAGE2_CAN_OVERTURN else s1.flag
    latency = c["latency_ms"] + (time.perf_counter() - t0) * 1000
    return MonitorResult(trace.id, s1, s2, final, tokens, latency)


def _try(fn, trace, label):
    """Retry twice with backoff, then log and skip."""
    for attempt in range(3):
        try:
            return fn(trace)
        except Exception as e:
            if attempt == 2:
                print(f"[{label}] skipping {trace.id}: {e!r}")
                return None
            time.sleep(2 * (attempt + 1))


def _append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_dict(obj)) + "\n")


def phase1(traces, deadline):
    done = {r["trace_id"] for r in read_jsonl(STAGE1_CACHE)}
    todo = [t for t in traces if t.id not in done]
    print(f"[stage1] {len(done)} cached, {len(todo)} to run -> {STAGE1_CACHE}")

    def run(t):
        t0 = time.perf_counter()
        s1, i, o = classify(t)
        return Stage1Cache(t.id, s1, i, o, (time.perf_counter() - t0) * 1000)

    for t in tqdm(todo, desc="stage1 (d3)"):
        if time.time() > deadline:
            print("[stage1] time budget reached, stopping; results so far are saved"); break
        r = _try(run, t, "stage1")
        if r:
            _append(STAGE1_CACHE, r)


def phase2(traces, deadline, baseline=True):
    cache = {r["trace_id"]: r for r in read_jsonl(STAGE1_CACHE)}
    done_c = {r["trace_id"] for r in read_jsonl(MONITOR_RESULTS)}
    done_b = {r["trace_id"] for r in read_jsonl(BASELINE_RESULTS)}
    missing = [t.id for t in traces if t.id not in cache]
    if missing:
        print(f"[stage2] {len(missing)} traces have no stage-1 result yet; cascade skips them")
    todo = [t for t in traces if (t.id in cache and t.id not in done_c) or (baseline and t.id not in done_b)]
    print(f"[stage2] cascade done {len(done_c)}, baseline done {len(done_b)}, {len(todo)} traces to touch")

    for t in tqdm(todo, desc="stage2 (MiniCPM)"):
        if time.time() > deadline:
            print("[stage2] time budget reached, stopping; results so far are saved"); break
        if t.id in cache and t.id not in done_c:
            r = _try(lambda tr: cascade_from_cache(tr, cache[tr.id]), t, "cascade")
            if r:
                _append(MONITOR_RESULTS, r)
        if baseline and t.id not in done_b:
            r = _try(analyze_explain_all, t, "explain-all")
            if r:
                _append(BASELINE_RESULTS, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["stage1", "stage2"])
    ap.add_argument("files", nargs="*", help="trace JSONL files (default: data/traces*.jsonl or fixtures)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    deadline = time.time() + args.minutes * 60 if args.minutes else float("inf")
    traces = load_traces(args.files or None)
    print(f"Loaded {len(traces)} traces")
    if args.phase == "stage1":
        phase1(traces, deadline)
    else:
        phase2(traces, deadline, baseline=not args.skip_baseline)


if __name__ == "__main__":
    main()
