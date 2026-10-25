"""Shared loaders for Part C (eval + demo)."""
import json
from pathlib import Path

from common.schemas import Trace

DATA = Path("data")
RESULTS = Path("results")
MONITOR_RESULTS = RESULTS / "monitor_results.jsonl"
BASELINE_RESULTS = RESULTS / "baseline_results.jsonl"


def default_trace_files() -> list[Path]:
    """Real data if Part A has shipped it, otherwise the fixtures."""
    real = [p for p in (DATA / "traces.jsonl", DATA / "traces_evasion.jsonl") if p.exists()]
    return real or [DATA / "sample_traces.jsonl"]


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] {path}:{i} is not valid JSON, skipping")
    return rows


def load_traces(paths=None) -> list[Trace]:
    """Load and dedupe traces (by id) from one or more JSONL files."""
    seen = {}
    for p in paths or default_trace_files():
        for row in read_jsonl(p):
            try:
                t = Trace(**row)
            except TypeError as e:
                print(f"[warn] bad trace in {p}: {e}")
                continue
            seen[t.id] = t
    return list(seen.values())
