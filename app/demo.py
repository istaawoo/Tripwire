"""Tattletale live demo.

    python -m app.demo            # uses monitor.pipeline (real models once Part B merges)
    python -m app.demo --offline  # keyword stub, no model calls
"""
import argparse

import gradio as gr

from app import stub
from common.schemas import Trace
from eval.loaders import BASELINE_RESULTS, MONITOR_RESULTS, load_traces, read_jsonl

TRACES = {t.id: t for t in load_traces()}
CASCADE = {r["trace_id"]: r for r in read_jsonl(MONITOR_RESULTS)}
BASELINE = {r["trace_id"]: r for r in read_jsonl(BASELINE_RESULTS)}
CUSTOM = "(paste your own CoT)"


def _choice_label(t: Trace) -> str:
    r = CASCADE.get(t.id)
    fooled = r is not None and t.label == "malicious" and not r["final_flag"]
    return f"{'[FOOLED MONITOR] ' if fooled else ''}{t.id}  ({t.label}, {t.difficulty})"


# fooled traces first so they're easy to find on stage
CHOICES = {_choice_label(t): t.id for t in sorted(
    TRACES.values(), key=lambda t: (not _choice_label(t).startswith("[FOOLED"), t.id))}


def pick(choice):
    t = TRACES.get(CHOICES.get(choice))
    if t is None:
        return "", "", "", ""
    truth = f"**Ground truth:** {t.label} · {t.category} · {t.difficulty} · source={t.source}"
    return t.task, t.cot, t.final_answer, truth


def _highlight(cot, span):
    if not span or span not in cot:
        return [(cot, None)]
    i = cot.index(span)
    return [(cot[:i], None), (span, "suspicious"), (cot[i + len(span):], None)]


def run(task, cot, final_answer, choice, offline):
    if not cot.strip():
        return "Paste a chain of thought first.", [], "", ""
    trace_id = CHOICES.get(choice, "custom")
    t = TRACES.get(trace_id)
    if t is None or t.cot != cot:  # edited or custom text
        t = Trace("custom", task, cot, final_answer, "benign", "none", "clean", "real")

    if offline:
        r = stub.analyze(t)
    else:
        try:
            from monitor.pipeline import analyze
            r = analyze(t)
        except Exception as e:
            r = stub.analyze(t)
            gr.Warning(f"Model call failed ({e!r}); showing offline stub result.")

    s1 = r.stage1
    verdict = "🚩 FLAGGED" if r.final_flag else "✅ Not flagged"
    if s1.flag and r.stage2 is not None and not r.stage2.confirmed:
        verdict = "↩️ Stage 1 flagged, stage 2 overturned"
    s1_md = (f"### {verdict}\n"
             f"**Stage 1:** flag={s1.flag} · category=`{s1.category}` · confidence={s1.confidence:.2f}\n\n"
             f"Latency: {r.latency_ms:.0f} ms")

    if r.stage2 is None:
        s2_md = "_Stage 2 skipped: stage 1 did not flag (this is where the tokens are saved)._"
    else:
        s2_md = (f"**Confirmed:** {r.stage2.confirmed}\n\n"
                 f"**Explanation:** {r.stage2.explanation}\n\n"
                 f"**Predicted next action:** {r.stage2.predicted_next_action}")

    tk = r.tokens
    casc_tokens = tk.stage1_in + tk.stage1_out + tk.stage2_in + tk.stage2_out
    base = BASELINE.get(t.id)
    base_tokens = sum(base["tokens"].values()) if base else None
    tok_md = (f"**Tokens (cascade):** {casc_tokens:,}  (s1 {tk.stage1_in}+{tk.stage1_out}, "
              f"s2 {tk.stage2_in}+{tk.stage2_out})\n\n"
              f"**Tokens (explain-all baseline):** "
              f"{f'{base_tokens:,}' if base_tokens is not None else 'n/a (not in baseline results)'}")
    return s1_md, _highlight(cot, s1.suspicious_span), s2_md, tok_md


def build(offline_default=False):
    with gr.Blocks(title="Tattletale: CoT Monitor") as demo:
        gr.Markdown("# 🕵️ Tattletale\nA cheap chain-of-thought monitor: stage 1 flags, stage 2 explains.")
        with gr.Row():
            with gr.Column():
                choice = gr.Dropdown([CUSTOM, *CHOICES], value=CUSTOM, label="Pick a trace")
                truth = gr.Markdown()
                task = gr.Textbox(label="Task", lines=2)
                cot = gr.Textbox(label="Chain of thought", lines=10)
                final = gr.Textbox(label="Final answer", lines=2)
                offline = gr.Checkbox(value=offline_default, label="Offline mode (keyword stub, no model calls)")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                s1_out = gr.Markdown()
                hl = gr.HighlightedText(label="CoT with suspicious span", color_map={"suspicious": "#eb6834"})
                s2_out = gr.Markdown()
                tok_out = gr.Markdown()
        choice.change(pick, choice, [task, cot, final, truth])
        btn.click(run, [task, cot, final, choice, offline], [s1_out, hl, s2_out, tok_out])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--share", action="store_true", help="public gradio link")
    args = ap.parse_args()
    build(args.offline).launch(share=args.share)
