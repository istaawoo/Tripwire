"""Run the monitor over trace files and write results (resume-safe).

  python -m eval.run                       # data/traces.jsonl + data/traces_evasion.jsonl (or sample fixtures)
  python -m eval.run data/sample_traces.jsonl

Writes results/monitor_results.jsonl (cascade) and results/baseline_results.jsonl (explain-all),
one MonitorResult per line. Trace ids already present are skipped.
"""
import json
import sys
from pathlib import Path

from tqdm import tqdm

from common.schemas import Trace, to_dict
from monitor.pipeline import analyze, analyze_explain_all

RESULTS = Path("results")
DEFAULT = ["data/traces.jsonl", "data/traces_evasion.jsonl"]


def load_traces(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.exists():
            out += [Trace(**json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["trace_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def run(traces, fn, path):
    done = done_ids(path)
    todo = [t for t in traces if t.id not in done]
    for t in tqdm(todo, desc=path.stem):
        try:
            r = fn(t)
        except Exception as e:
            print(f"SKIP {t.id}: {e!r:.200}", file=sys.stderr)
            continue
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_dict(r)) + "\n")


def main(argv):
    paths = argv or DEFAULT
    traces = load_traces(paths)
    if not traces:
        traces = load_traces(["data/sample_traces.jsonl"])
    print(f"{len(traces)} traces")
    RESULTS.mkdir(exist_ok=True)
    run(traces, analyze, RESULTS / "monitor_results.jsonl")
    run(traces, analyze_explain_all, RESULTS / "baseline_results.jsonl")


if __name__ == "__main__":
    main(sys.argv[1:])
