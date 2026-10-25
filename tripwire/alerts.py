"""Trip rules. A question trips when its risk crosses THRESHOLD, or jumps by more
than JUMP since the previous check. Each question trips at most once per session.
Evidence = the steps added since the previous check (the ones that moved it)."""
from .checklist import QUESTION_NAMES

THRESHOLD = 0.7
JUMP = 0.3


def find_trips(checks, threshold=THRESHOLD, jump=JUMP):
    trips, prev, fired = [], {}, set()
    for c in checks:
        for q in QUESTION_NAMES:
            p = c["risk"][q]
            before = prev.get(q)
            reason = None
            if p >= threshold:
                reason = f"p={p:.2f} >= {threshold}"
            elif before is not None and p - before > jump:
                reason = f"jump {before:.2f} -> {p:.2f}"
            if reason and q not in fired:
                fired.add(q)
                trips.append({"question": q, "step": c["step"], "p": p, "prev": before,
                              "reason": reason, "evidence": c["new"]})
            prev[q] = p
    return trips
