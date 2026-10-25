"""Tripwire replay UI.   streamlit run app.py"""
import json
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from tripwire import backends, monitor
from tripwire.checklist import QUESTION_NAMES, SHORT

st.set_page_config(page_title="Tripwire", layout="wide")
ROOT = Path(__file__).parent
BADGE = {"FAKE": "#d33", "d3": "#1a7f37", "fallback": "#b08800"}


def badge(label):
    color = BADGE["FAKE"] if label == "FAKE" else BADGE["d3"] if label.startswith("d3") else BADGE["fallback"]
    note = "  (UI demo numbers, not results)" if label == "FAKE" else ""
    st.markdown(f"<span style='background:{color};color:white;padding:4px 10px;border-radius:6px;font-weight:600'>"
                f"backend: {label}</span><span style='color:{color}'>{note}</span>", unsafe_allow_html=True)


def chart(run, upto):
    checks = [c for c in run["checks"] if c["step"] <= upto]
    fig = go.Figure()
    xs = [c["step"] for c in checks]
    for q in QUESTION_NAMES:
        fig.add_trace(go.Scatter(x=xs, y=[c["risk"][q] for c in checks], mode="lines", name=SHORT[q]))
    trips = [t for t in run["trips"] if t["step"] <= upto]
    if trips:
        fig.add_trace(go.Scatter(x=[t["step"] for t in trips], y=[t["p"] for t in trips], mode="markers+text",
                                 marker=dict(color="red", size=16, symbol="x"), text=[SHORT[t["question"]] for t in trips],
                                 textposition="top center", name="TRIP"))
    fig.add_hline(y=0.7, line_dash="dot", line_color="red")
    fig.update_layout(height=420, yaxis=dict(range=[0, 1.05], title="P(bad)"),
                      xaxis=dict(range=[0, len(run["steps"])], title="step"), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def msg_html(s, hot=False):
    bg = {"user": "#e8f0fe", "assistant": "#f6f8fa", "tool": "#fff8e1"}.get(s["role"], "#fff")
    border = "3px solid red" if hot else "1px solid #ddd"
    body = s["text"].replace("&", "&amp;").replace("<", "&lt;")[:1500]
    return (f"<div style='background:{bg};border:{border};border-radius:6px;padding:6px 10px;margin:4px 0;"
            f"font-family:monospace;font-size:12px;white-space:pre-wrap;color:#111'><b>[{s['i']} {s['role']}]</b>\n{body}</div>")


st.title("Tripwire")
st.caption("A small decision model watches an AI agent's transcript and trips the moment it tries something it shouldn't.")

runs = sorted(ROOT.glob("runs/*/*.json"))
mode = st.sidebar.radio("Source", ["Saved run", "Live (fake/ollama)"])
if mode == "Saved run":
    if not runs:
        st.warning("No saved runs in runs/. Create one: python -m tripwire run transcripts --backend fake --out runs/fake")
        st.stop()
    pick = st.sidebar.selectbox("Run", runs, format_func=lambda p: f"{p.parent.name} / {p.stem}")
    run = json.loads(pick.read_text())
else:
    files = sorted(ROOT.glob("transcripts/*.jsonl")) + sorted(ROOT.glob("examples/*.jsonl"))
    f = st.sidebar.selectbox("Transcript", files, format_func=lambda p: p.name)
    spec = st.sidebar.text_input("Backend", "fake")
    lp = ROOT / "transcripts/labels.json"
    labels = json.loads(lp.read_text()) if lp.exists() else []
    run = monitor.run(f, backends.make(spec, labels))
speed = st.sidebar.slider("Seconds per check", 0.1, 2.0, 0.6)

badge(run["backend"])
left, right = st.columns([3, 2])
plot, trip_box = left.empty(), left.empty()
right.subheader(run["file"])
feed = right.container(height=560)

if st.sidebar.button("▶ Play", type="primary"):
    shown = -1
    for c in run["checks"]:
        for i in c["new"]:
            feed.markdown(msg_html(run["steps"][i], hot=any(i in t["evidence"] for t in run["trips"] if t["step"] == c["step"])),
                          unsafe_allow_html=True)
        plot.plotly_chart(chart(run, c["step"]), use_container_width=True, key=f"p{c['step']}")
        new = [t for t in run["trips"] if t["step"] == c["step"]]
        if new:
            trip_box.error(" | ".join(f"TRIP {t['question']} at step {t['step']} ({t['reason']})" for t in new))
        time.sleep(speed)
    st.session_state["played"] = run["file"]
else:
    plot.plotly_chart(chart(run, len(run["steps"])), use_container_width=True)

if run["trips"]:
    st.subheader("Trips: what caused them")
    sel = st.selectbox("Inspect trip", run["trips"],
                       format_func=lambda t: f"{t['question']} @ step {t['step']}  ({t['reason']})")
    st.caption("The decision model does not write explanations. Evidence = the steps added since the previous check, which moved the probability.")
    for i in sel["evidence"]:
        st.markdown(msg_html(run["steps"][i], hot=True), unsafe_allow_html=True)
else:
    st.success("No trips in this session.")
