"""Load agent transcripts into a flat list of steps.

Supported JSONL formats (auto-detected per line):
  1. simple:      {"role": "user|assistant|tool", "content": "..."}
  2. OpenAI:      chat messages incl. assistant "tool_calls" and role "tool",
                  one per line, or one line {"messages": [...]}
  3. Claude Code: session JSONL lines {"type": "user|assistant", "message": {...}}
                  with content blocks text / tool_use / tool_result.
Each step: {"i": int, "role": str, "text": str}.
"""
import json
from pathlib import Path


def _text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text_of(c.get("text", c.get("content")) if isinstance(c, dict) else c) for c in content)
    return str(content)


def _claude_blocks(role, content):
    if isinstance(content, str):
        yield role, content
        return
    for b in content or []:
        t = b.get("type")
        if t == "text":
            yield role, b.get("text", "")
        elif t == "tool_use":
            inp = b.get("input", {})
            cmd = inp.get("command") if isinstance(inp, dict) else None
            yield "assistant", f"[tool {b.get('name')}] " + (f"$ {cmd}" if cmd else json.dumps(inp))
        elif t == "tool_result":
            yield "tool", _text_of(b.get("content"))
        # thinking / images skipped


def _openai_msg(m):
    role = m.get("role", "assistant")
    out = []
    txt = _text_of(m.get("content"))
    if txt:
        out.append((role, txt))
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        out.append(("assistant", f"[tool {fn.get('name')}] {fn.get('arguments', '')}"))
    return out


def parse_lines(lines):
    pairs = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if "messages" in obj and isinstance(obj["messages"], list):
            for m in obj["messages"]:
                pairs += _openai_msg(m)
        elif obj.get("type") in ("user", "assistant") and isinstance(obj.get("message"), dict):
            msg = obj["message"]
            pairs += list(_claude_blocks(msg.get("role", obj["type"]), msg.get("content")))
        elif "role" in obj:
            pairs += _openai_msg(obj)
        # other Claude Code line types (summary, system, etc.) are skipped
    return [{"i": i, "role": r, "text": t} for i, (r, t) in enumerate(pairs)]


def load(path):
    return parse_lines(Path(path).read_text(encoding="utf-8").splitlines())


def render(steps, max_chars=8000, mark_from=None):
    """Render steps as monitor input. Steps with i >= mark_from are tagged NEW."""
    parts = []
    for s in steps:
        tag = " NEW" if mark_from is not None and s["i"] >= mark_from else ""
        parts.append(f"[step {s['i']} {s['role']}{tag}]\n{s['text']}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:  # keep the tail: newest steps matter most
        text = "...\n" + text[-max_chars:]
    return text
