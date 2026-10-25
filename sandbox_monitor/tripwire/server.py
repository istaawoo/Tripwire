#!/usr/bin/env python3
"""Live demo server for the Tripwire monitor.

Keeps d3 and MiniCPM5 resident in MLX and streams every step of the pipeline to
the web UI as server-sent events: canary scan, each d3 answer as it lands, the
policy decision, then the MiniCPM5 summary token by token.

Every successful live run is recorded to logs/replay/<scenario>.json, and
`--record` pre-records all scenarios. Replay mode plays those recordings back
(clearly labelled in the UI) in case a live call misbehaves on stage.

Usage:
  python -m tripwire.server            # http://localhost:8765
  python -m tripwire.server --record   # record replays for every scenario, then exit
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tripwire import monitor as M

ROOT = M.ROOT
WEB = ROOT / "web"
REPLAY = ROOT / "logs" / "replay"
PORT = 8765
NOUL = M.TERMINATE_CLASSES + ["flag"]
MAX_REPLAY_GAP = 2.5  # seconds; replay never sits idle longer than this between events

LOCK = threading.Lock()  # one analysis at a time
JOBS: queue.Queue = queue.Queue()  # MLX is not thread-safe: one worker thread owns every model call
STATE = {"ready": False, "error": None}
D3 = SUMM = TOK = None


def load_models() -> None:
    global D3, SUMM, TOK
    try:
        from mlx_lm import load, stream_generate
        from tripwire.d3_mlx import D3 as D3Model
        t = time.time()
        D3 = D3Model()
        D3.ask("warm-up", {"type": "noul", "instructions": "Is this a test?"})
        SUMM, TOK = load(M.SUMMARIZER)
        for _ in stream_generate(SUMM, TOK, "hi", max_tokens=2):
            pass
        STATE["ready"] = True
        print(f"[tripwire] models ready in {time.time() - t:.1f}s -> http://localhost:{PORT}", flush=True)
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()


def mlx_worker() -> None:
    """Loads the models, then runs every analysis, all on this one thread."""
    load_models()
    while True:
        transcript, out, cancel = JOBS.get()
        try:
            for ev in analyze(transcript):
                out.put(ev)
                if cancel.is_set():
                    break
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            out.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            out.put(None)


def scenarios() -> list[dict]:
    out = []
    for d in sorted(p for p in (ROOT / "scenarios").iterdir() if p.is_dir()):
        text = (d / "transcript.txt").read_text()
        exp = json.loads((d / "expected.json").read_text()) if (d / "expected.json").exists() else {}
        out.append({"id": d.name, "transcript": M.redact(text), "hits": M.canary_hits(text),
                    "expected": exp, "has_replay": (REPLAY / f"{d.name}.json").exists()})
    return out


def analyze(transcript: str):
    """Yield pipeline events. Same order and policy as tripwire.monitor."""
    from mlx_lm import stream_generate
    t0 = time.time()
    hits = M.canary_hits(transcript)
    yield {"type": "canary", "hits": hits}
    if hits:  # binding: the run is killed before d3 is even asked
        yield {"type": "decision", "action": "TERMINATE", "early": True,
               "reason": f"canary (key.txt contents) found at line {hits[0]['line']}"}

    state = M.CONTEXT + transcript
    probs = {}
    for name, q in M.QUESTIONS.items():
        t = time.time()
        ans = D3.ask(state, q)
        ev = {"type": "question", "name": name, "ms": round((time.time() - t) * 1000)}
        if q["type"] == "noul":
            probs[name] = ans["noul"]
            ev["p"] = ans["noul"]
        else:
            ev.update(choice=ans["choice"], confidence=ans["confidence"], probabilities=ans["probabilities"])
        yield ev

    action, reason = M.decide(probs, hits)
    yield {"type": "decision", "action": action, "reason": reason, "early": False}

    if action != "CONTINUE":
        yield {"type": "summary_start"}
        prompt = M.summary_prompt(TOK, transcript, action, reason)
        text, shown = "", 0
        hold = max(len(c) for c in M.CANARIES)  # a secret can't stream out half-written
        for r in stream_generate(SUMM, TOK, prompt, max_tokens=500):
            text += r.text
            if "<think>" in text and "</think>" not in text:
                continue  # never stream the model's thinking
            visible = M.redact(text.split("</think>")[-1]).lstrip()
            safe = len(visible) - hold
            if safe > shown:
                yield {"type": "summary_token", "text": visible[shown:safe]}
                shown = safe
        final = M.redact(text.split("</think>")[-1].strip())
        yield {"type": "summary_done", "text": final}
    yield {"type": "done", "total_ms": round((time.time() - t0) * 1000)}


def record(scenario_id: str, events: list[tuple[float, dict]]) -> None:
    REPLAY.mkdir(parents=True, exist_ok=True)
    (REPLAY / f"{scenario_id}.json").write_text(json.dumps(
        {"scenario": scenario_id, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "events": [{"t": round(t, 3), "event": e} for t, e in events]}, indent=1))


def replay(scenario_id: str):
    data = json.loads((REPLAY / f"{scenario_id}.json").read_text())
    last = 0.0
    for item in data["events"]:
        time.sleep(min(max(item["t"] - last, 0.0), MAX_REPLAY_GAP))
        last = item["t"]
        yield item["event"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: only errors reach the terminal
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (WEB / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._json({**STATE, "busy": LOCK.locked()})
        elif self.path == "/api/scenarios":
            self._json(scenarios())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/run":
            return self._json({"error": "not found"}, 404)
        req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        mode, sid = req.get("mode", "live"), req.get("scenario")

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-store")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(ev):
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()

        try:
            if mode == "replay":
                if not sid or not (REPLAY / f"{sid}.json").exists():
                    return send({"type": "error", "message": "No recording for this scenario yet."})
                for ev in replay(sid):
                    send(ev)
                return
            if not STATE["ready"]:
                return send({"type": "error", "message": STATE["error"] or "Models are still loading."})
            if sid:
                transcript = (ROOT / "scenarios" / sid / "transcript.txt").read_text()
            else:
                transcript = req.get("transcript", "")
            if not transcript.strip():
                return send({"type": "error", "message": "Transcript is empty."})
            if not LOCK.acquire(timeout=60):  # a cancelled run winds down within a question or two
                return send({"type": "error", "message": "Another analysis is running. Wait for it to finish."})
            out, cancel = queue.Queue(), threading.Event()
            try:
                JOBS.put((transcript, out, cancel))
                t0, events = time.time(), []
                while (ev := out.get()) is not None:
                    events.append((time.time() - t0, ev))
                    try:
                        send(ev)
                    except OSError:
                        cancel.set()  # browser went away: stop the worker, drain, release
                        while out.get() is not None:
                            pass
                        return
                if sid and events and events[-1][1]["type"] == "done":
                    record(sid, events)
            finally:
                LOCK.release()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away; the next run starts clean
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            except OSError:
                pass


def record_all() -> int:
    load_models()
    if not STATE["ready"]:
        return 1
    for s in scenarios():
        transcript = (ROOT / "scenarios" / s["id"] / "transcript.txt").read_text()
        t0, events = time.time(), []
        for ev in analyze(transcript):
            events.append((time.time() - t0, ev))
        record(s["id"], events)
        dec = next(e for _, e in reversed(events) if e["type"] == "decision")
        ok = dec["action"] == s["expected"].get("expected_action", "").upper()
        print(f"[record] {s['id']:<26} {dec['action']:<10} {'ok' if ok else 'MISMATCH'}  {events[-1][0]:.1f}s",
              flush=True)
    return 0


def main(argv: list[str]) -> int:
    if "--record" in argv:
        return record_all()
    threading.Thread(target=mlx_worker, daemon=True).start()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"[tripwire] port {PORT} is busy: Tripwire is probably already running. "
              f"Open http://localhost:{PORT}, or stop it with: pkill -f tripwire.server")
        return 1
    print(f"[tripwire] serving http://localhost:{PORT} (models loading...)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
