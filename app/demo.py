"""Tattletale demo.   python -m app.demo   ->  http://127.0.0.1:7860

Pick a trace (or paste a CoT) and see the cascade as a timeline:
CoT sentences in order, shaded by stage-1 suspicion -> stage 1 verdict -> stage 2 verdict.
Reads saved results so it works offline; "Run live" calls monitor.pipeline.analyze (stub works too).
"""
import html
import json
import re
from pathlib import Path

import gradio as gr

DATA, RESULTS = Path("data"), Path("results")
RED, BLUE, ORANGE, GREEN, INK, MUTED, LINE = "#e34948", "#2a78d6", "#eb6834", "#008300", "#0b0b0b", "#52514e", "#e6e5e0"


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def sentences(cot):
    try:  # use B's exact splitter so sentence ids line up with d3's span answer
        from monitor.stage1_d3 import sentences as s
        return s(cot)
    except Exception:
        return [m.group(0).strip() for m in re.finditer(r"[^.!?\n]+[.!?]*", cot) if m.group(0).strip()]


def load_all():
    traces = {}
    for f in sorted(DATA.glob("*.jsonl")):
        for t in read_jsonl(f):
            traces[t["id"]] = t
    cas = {r["trace_id"]: r for r in read_jsonl(RESULTS / "monitor_results.jsonl")}
    base = {r["trace_id"]: r for r in read_jsonl(RESULTS / "baseline_results.jsonl")}
    d3 = {r["trace_id"]: r for r in read_jsonl(RESULTS / "stage1_d3_cache.jsonl")}
    return traces, cas, base, d3


TRACES, CAS, BASE, D3 = load_all()


def outcome(t, r):
    if r is None:
        return "not run"
    mal, flag = t["label"] == "malicious", r["final_flag"]
    return {(True, True): "caught", (True, False): "MISSED", (False, True): "false alarm", (False, False): "correctly clear"}[(mal, flag)]


def choices():
    out = []
    for i, t in TRACES.items():
        o = outcome(t, CAS.get(i))
        tag = "  ★ fooled the monitor" if o in ("MISSED", "false alarm") else ""
        out.append((f"{i}  [{t['label']}, {t['difficulty']}] -> {o}{tag}", i))
    return sorted(out, key=lambda c: ("★" not in c[0], c[1]))


def tok_total(r):
    t = r["tokens"]
    return t["stage1_in"] + t["stage1_out"] + t["stage2_in"] + t["stage2_out"]


def timeline_html(t, r, d3row):
    cot = t["cot"]
    sents = sentences(cot)
    span = (r or {}).get("stage1", {}).get("suspicious_span", "")
    probs = (d3row or {}).get("probs", {}).get("span", {}) if d3row else {}
    total = sum(probs.values()) or 1
    cells = []
    for k, s in enumerate(sents):
        p = probs.get(f"s{k}", 0) / total if probs else (1.0 if span and (s in span or span in s) else 0.0)
        alpha = 0.08 + 0.82 * p
        hot = span and (s in span or span in s)
        cells.append(
            f"<div title='suspicion {p:.0%}' style='flex:1 1 0;min-width:28px;height:34px;border-radius:4px;"
            f"background:rgba(227,73,72,{alpha:.2f});border:{'2px solid ' + RED if hot else '1px solid ' + LINE};"
            f"display:flex;align-items:center;justify-content:center;font:600 11px system-ui;color:{INK}'>{k + 1}</div>")
    strip = "<div style='display:flex;gap:3px;margin:6px 0 2px'>" + "".join(cells) + "</div>"
    legend = (f"<div style='font:12px system-ui;color:{MUTED}'>Each block is one sentence of the chain of thought, in order. "
              f"Darker red = stage 1 rated it more suspicious{' (d3-flash per-sentence probability)' if probs else ''}. "
              f"Outlined = the quoted suspicious span.</div>")
    body = html.escape(cot)
    if span and span in cot:
        e = html.escape(span)
        body = body.replace(e, f"<mark style='background:#fbd5d4;color:{INK};padding:1px 2px;border-radius:3px'>{e}</mark>", 1)
    return strip + legend + (f"<div style='margin-top:10px;padding:10px 12px;border:1px solid {LINE};border-radius:8px;"
                             f"font:13px/1.55 ui-monospace,monospace;white-space:pre-wrap;color:{INK};background:#fcfcfb'>{body}</div>")


def card(title, color, inner):
    return (f"<div style='flex:1;min-width:200px;border:1px solid {LINE};border-top:4px solid {color};border-radius:8px;"
            f"padding:10px 12px;background:#fff;color:{INK};font:13px/1.5 system-ui'><div style='font-weight:700;margin-bottom:4px'>{title}</div>{inner}</div>")


def cascade_html(t, r, b):
    if r is None:
        return f"<div style='color:{MUTED};font:13px system-ui'>No saved result for this trace. Click Run live.</div>"
    s1, s2 = r["stage1"], r["stage2"]
    c1 = card("Stage 1 · decision model", RED if s1["flag"] else GREEN,
              f"<b>{'FLAG' if s1['flag'] else 'pass'}</b> · {s1['category']}<br>confidence {s1['confidence']:.0%}"
              f"<div style='height:6px;background:{LINE};border-radius:3px;margin-top:4px'><div style='height:6px;width:{s1['confidence'] * 100:.0f}%;"
              f"background:{RED if s1['flag'] else GREEN};border-radius:3px'></div></div>")
    if s2 is None:
        c2 = card("Stage 2 · explainer", LINE, "Skipped: stage 1 did not flag, so no expensive call.")
    else:
        c2 = card("Stage 2 · explainer", RED if s2["confirmed"] else GREEN,
                  f"<b>{'CONFIRMED' if s2['confirmed'] else 'OVERTURNED'}</b><br>{html.escape(s2['explanation'])}"
                  f"<br><span style='color:{MUTED}'>Likely next: {html.escape(s2['predicted_next_action'])}</span>")
    o = outcome(t, r)
    oc = {"caught": GREEN, "correctly clear": GREEN, "MISSED": RED, "false alarm": ORANGE}.get(o, MUTED)
    truth = f"truth: {t['label']} ({t['category']}, {t['difficulty']})"
    toks = f"cascade {tok_total(r):,} tokens" + (f" vs explain-all {tok_total(b):,}" if b else "")
    c3 = card("Final", oc, f"<b>{'ALERT' if r['final_flag'] else 'no alert'}</b> · <b style='color:{oc}'>{o}</b><br>{truth}<br>"
                           f"<span style='color:{MUTED}'>{toks} · {r['latency_ms']:.0f} ms</span>")
    arrow = f"<div style='align-self:center;color:{MUTED};font-size:20px'>→</div>"
    return f"<div style='display:flex;gap:8px;flex-wrap:wrap'>{c1}{arrow}{c2}{arrow}{c3}</div>"


def show(tid):
    if not tid or tid not in TRACES:
        return "", ""
    t = TRACES[tid]
    return timeline_html(t, CAS.get(tid), D3.get(tid)), cascade_html(t, CAS.get(tid), BASE.get(tid))


def run_live(task, cot, final):
    from common.schemas import Trace, to_dict
    from monitor.pipeline import analyze
    t = Trace(id="live", task=task or "(none)", cot=cot or "", final_answer=final or "", label="benign",
              category="none", difficulty="clean", source="real")
    r = to_dict(analyze(t))
    td = to_dict(t) | {"label": "?", "category": "?", "difficulty": "pasted"}
    return timeline_html(td, r, None), cascade_html(td | {"label": "unknown"}, r, None).replace("correctly clear", "clear")


def metrics_md():
    p = RESULTS / "metrics.json"
    if not p.exists():
        return "No results/metrics.json yet. Run `python -m eval.metrics`."
    m = json.loads(p.read_text())
    r, f, c = m["1_recall_by_difficulty"], m["2_false_positive_rate"], m["3_cascade_vs_explain_all"]
    pct = lambda x: "n/a" if x["rate"] is None else f"{x['rate']:.0%} ({x['hits']}/{x['n']})"
    rows = [
        "| metric | value |", "|---|---|",
        f"| Detection, obvious | {pct(r['obvious'])} |", f"| Detection, subtle | {pct(r['subtle'])} |",
        f"| False alarms, hard negatives | {pct(f['hard_negative'])} |", f"| False alarms, clean | {pct(f['clean'])} |",
        f"| Expensive-model tokens, cascade vs explain-all | {c['cascade']['tokens']['stage2_expensive']:,} vs {c['explain_all']['tokens']['stage2_expensive']:,} |",
        f"| Recall, cascade vs explain-all | {pct(c['cascade']['recall'])} vs {pct(c['explain_all']['recall'])} |",
        f"| Detection before vs after evasion | {pct(m['4_evasion']['originals'])} vs {pct(m['4_evasion']['evasion_r1'])} |",
        f"| Stage 2 overturn rate | {pct(m['5_stage2_overturn_rate'])} |",
        f"| Stage 1 alone recall | {pct(m['6_stage1_alone_recall'])} |"]
    return "\n".join(rows)


with gr.Blocks(title="Tattletale", theme=gr.themes.Default()) as demo:
    gr.Markdown("# Tattletale\nA cheap decision model reads every chain of thought; an explainer model only looks at what it flags.")
    with gr.Tab("Trace timeline"):
        pick = gr.Dropdown(choices=choices(), label="Trace (★ = fooled the monitor)", value=(choices() or [(None, None)])[0][1])
        tl, cs = gr.HTML(), gr.HTML()
        pick.change(show, pick, [tl, cs])
        demo.load(show, pick, [tl, cs])
    with gr.Tab("Paste a CoT (live)"):
        task = gr.Textbox(label="Task", lines=1)
        cot = gr.Textbox(label="Chain of thought", lines=8)
        fin = gr.Textbox(label="Final answer", lines=2)
        go = gr.Button("Run live", variant="primary")
        tl2, cs2 = gr.HTML(), gr.HTML()
        go.click(run_live, [task, cot, fin], [tl2, cs2])
    with gr.Tab("Results"):
        gr.Markdown(metrics_md())
        imgs = [str(RESULTS / f) for f in ("cascade_vs_explain_all.png", "detection.png", "evasion.png", "recall_by_category.png")
                if (RESULTS / f).exists()]
        if imgs:
            gr.Gallery(imgs, columns=1, height="auto", show_label=False)

if __name__ == "__main__":
    demo.launch()
