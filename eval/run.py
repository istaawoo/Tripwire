"""Run the cascade and the explain-all baseline over traces. Resume-safe.

    python -m eval.run                          # data/traces*.jsonl, else fixtures
    python -m eval.run data/sample_traces.jsonl
    python -m eval.run --skip-baseline          # cascade only (saves time)
    python -m eval.run --fresh                  # wipe results and start over
"""
import argparse
import json
import time

from tqdm import tqdm

from common.schemas import to_dict
from eval.loaders import BASELINE_RESULTS, MONITOR_RESULTS, RESULTS, load_traces, read_jsonl
from monitor.pipeline import analyze, analyze_explain_all


def run(fn, traces, out_path, label):
    done = {r["trace_id"] for r in read_jsonl(out_path)}
    todo = [t for t in traces if t.id not in done]
    print(f"[{label}] {len(done)} already done, {len(todo)} to run -> {out_path}")
    failed = []
    with open(out_path, "a", encoding="utf-8") as f:
        for t in tqdm(todo, desc=label):
            for attempt in range(3):
                try:
                    r = fn(t)
                    f.write(json.dumps(to_dict(r)) + "\n")
                    f.flush()
                    break
                except Exception as e:  # retry twice, then log and skip
                    if attempt == 2:
                        print(f"[{label}] skipping {t.id}: {e!r}")
                        failed.append(t.id)
                    else:
                        time.sleep(2 * (attempt + 1))
    if failed:
        print(f"[{label}] {len(failed)} failed: {failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="trace JSONL files (default: data/traces*.jsonl or fixtures)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete existing results first")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    if args.fresh:
        for p in (MONITOR_RESULTS, BASELINE_RESULTS):
            p.unlink(missing_ok=True)

    traces = load_traces(args.files or None)
    print(f"Loaded {len(traces)} traces")
    run(analyze, traces, MONITOR_RESULTS, "cascade")
    if not args.skip_baseline:
        run(analyze_explain_all, traces, BASELINE_RESULTS, "explain-all")


if __name__ == "__main__":
    main()
