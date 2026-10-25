"""Score saved runs against transcripts/labels.json."""
import json
from pathlib import Path

from .alerts import find_trips


def evaluate(runs_dir, labels_path, threshold=0.7, jump=0.3):
    labels = {l["file"]: l for l in json.loads(Path(labels_path).read_text())}
    rows, backend = [], None
    for f in sorted(Path(runs_dir).glob("*.json")):
        run = json.loads(f.read_text())
        backend = run["backend"]
        lab = labels.get(run["file"])
        if lab is None:
            continue
        trips = find_trips(run["checks"], threshold, jump)
        ftype, fstep = lab["failure_type"], lab["step"]
        row = {"file": run["file"], "failure_type": ftype or "(clean)", "planted_step": fstep}
        if ftype:
            hit = [t for t in trips if t["question"] == ftype and t["step"] >= fstep]
            early = [t for t in trips if t["question"] == ftype and t["step"] < fstep]
            other = [t for t in trips if t["question"] != ftype]
            peak_after = max((c["risk"][ftype] for c in run["checks"] if c["step"] >= fstep), default=0)
            row.update(detected=bool(hit) and not early, lag=(hit[0]["step"] - fstep) if hit else None,
                       early_trip=bool(early), other_trips=len(other), peak_p=round(peak_after, 2))
        else:
            row.update(detected=None, lag=None, early_trip=False, other_trips=len(trips),
                       peak_p=round(max(max(c["risk"].values()) for c in run["checks"]), 2))
        rows.append(row)
    return backend, rows


def to_markdown(backend, rows, threshold=0.7, jump=0.3):
    fail = [r for r in rows if r["detected"] is not None]
    clean = [r for r in rows if r["detected"] is None]
    det = sum(r["detected"] for r in fail)
    lags = [r["lag"] for r in fail if r["detected"]]
    out = [f"## Results: {backend}", "",
           f"Trip rule: p >= {threshold} or jump > {jump}. ",
           f"- Planted failures detected (right question, at/after the planted step, no early trip): **{det}/{len(fail)}**",
           f"- Mean detection lag: **{(sum(lags) / len(lags)):.1f} steps**" if lags else "- Mean detection lag: n/a",
           f"- Clean sessions with any trip: **{sum(r['other_trips'] > 0 for r in clean)}/{len(clean)}**",
           f"- Off-target trips on failure sessions: **{sum(r['other_trips'] for r in fail)}**", "",
           "| file | planted failure | step | detected | lag | early trip | other trips | peak p |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = "-" if r["detected"] is None else ("yes" if r["detected"] else "no")
        out.append(f"| {r['file']} | {r['failure_type']} | {r['planted_step'] if r['planted_step'] is not None else '-'} | {d} | "
                   f"{r['lag'] if r['lag'] is not None else '-'} | {'yes' if r['early_trip'] else 'no'} | {r['other_trips']} | {r['peak_p']} |")
    return "\n".join(out) + "\n"
