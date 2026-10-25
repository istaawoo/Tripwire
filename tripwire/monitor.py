"""Replay a transcript through a backend: every N steps, show the model a sliding
window of recent steps and record a probability for every checklist answer."""
import time
from pathlib import Path

from .alerts import find_trips
from .checklist import QUESTION_NAMES, risk_of
from .transcript import load, render


def run(path, backend, every=2, window=20, max_chars=8000, on_check=None):
    steps = load(path)
    name = Path(path).name
    checks, last = [], -1
    for k in range(len(steps)):
        if (k + 1) % every and k != len(steps) - 1:
            continue
        lo = max(0, k - window + 1)
        state = render(steps[lo:k + 1], max_chars=max_chars, mark_from=last + 1)
        t0 = time.time()
        dist = backend.score(state, {"file": name, "step": k})
        chk = {
            "step": k,
            "new": list(range(last + 1, k + 1)),
            "secs": round(time.time() - t0, 2),
            "dist": dist,
            "risk": {q: round(risk_of(q, dist[q]), 4) for q in QUESTION_NAMES},
        }
        checks.append(chk)
        last = k
        if on_check:
            on_check(chk, steps)
    return {
        "file": name,
        "backend": backend.label,
        "every": every,
        "window": window,
        "steps": steps,
        "checks": checks,
        "trips": find_trips(checks),
    }
