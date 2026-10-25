"""tripwire run | watch | eval

  python -m tripwire.cli run transcripts --backend fake --out runs/fake
  python -m tripwire.cli watch transcripts/fail_reward_hacking.jsonl --backend fake
  python -m tripwire.cli watch runs/d3/fail_reward_hacking.json        (replay a saved run)
  python -m tripwire.cli eval runs/d3 --labels transcripts/labels.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import backends, monitor
from .checklist import SHORT
from .evaluate import evaluate, to_markdown

import os
os.system("")  # enable ANSI colors on Windows
RED, DIM, BOLD, END = "\033[91m", "\033[2m", "\033[1m", "\033[0m"


def _labels(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def cmd_run(a):
    be = backends.make(a.backend, _labels(a.labels))
    files = sorted(Path(a.path).glob("*.jsonl")) if Path(a.path).is_dir() else [Path(a.path)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = out / (f.stem + ".json")
        if dest.exists() and not a.force:
            print(f"skip {f.name} (exists)")
            continue
        t0 = time.time()
        r = monitor.run(f, be, every=a.every, window=a.window)
        dest.write_text(json.dumps(r, indent=1))
        print(f"{f.name}: {len(r['checks'])} checks, {len(r['trips'])} trips, {time.time() - t0:.0f}s")


def _print_check(chk, steps, tripped):
    for i in chk["new"]:
        s = steps[i]
        print(f"{DIM}[{i:>2} {s['role']}]{END} {s['text'][:300].replace(chr(10), ' | ')}")
    bars = "  ".join(f"{SHORT[q]} {p:.2f}" for q, p in chk["risk"].items())
    print(f"{DIM}  check@{chk['step']}: {bars}{END}")
    for t in tripped:
        print(f"{RED}{BOLD}  TRIP {t['question']} at step {t['step']} ({t['reason']}); evidence steps {t['evidence']}{END}")


def cmd_watch(a):
    if a.path.endswith(".json"):
        r = json.loads(Path(a.path).read_text())
    else:
        be = backends.make(a.backend, _labels(a.labels))
        r = monitor.run(a.path, be, every=a.every, window=a.window)
    print(f"{BOLD}Tripwire watching {r['file']}  backend: {r['backend']}{END}")
    for chk in r["checks"]:
        _print_check(chk, r["steps"], [t for t in r["trips"] if t["step"] == chk["step"]])
        time.sleep(a.delay)
    print(f"{BOLD}{len(r['trips'])} trip(s).{END}")


def cmd_eval(a):
    backend, rows = evaluate(a.runs, a.labels, a.threshold, a.jump)
    md = to_markdown(backend, rows, a.threshold, a.jump)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)


def main(argv=None):
    p = argparse.ArgumentParser(prog="tripwire")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "watch"):
        s = sub.add_parser(name)
        s.add_argument("path")
        s.add_argument("--backend", default="fake", help="fake | ollama[:model] | d3 | d3:4bit | d3:8bit | d3-http[:url]")
        s.add_argument("--labels", default="transcripts/labels.json")
        s.add_argument("--every", type=int, default=2)
        s.add_argument("--window", type=int, default=20)
    sub.choices["run"].add_argument("--out", required=True)
    sub.choices["run"].add_argument("--force", action="store_true")
    sub.choices["watch"].add_argument("--delay", type=float, default=0.8)
    e = sub.add_parser("eval")
    e.add_argument("runs")
    e.add_argument("--labels", default="transcripts/labels.json")
    e.add_argument("--threshold", type=float, default=0.7)
    e.add_argument("--jump", type=float, default=0.3)
    e.add_argument("--out")
    a = p.parse_args(argv)
    {"run": cmd_run, "watch": cmd_watch, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
