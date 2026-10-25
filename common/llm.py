import json
import re
from typing import Literal

from openai import OpenAI

from common.config import ROLES

_clients = {}


def _client(role):
    if role not in _clients:
        base_url, _, api_key = ROLES[role]
        _clients[role] = OpenAI(base_url=base_url, api_key=api_key)
    return _clients[role]


def chat(role: Literal["generator", "stage1", "stage2"], messages, json_mode=False, max_tokens=512) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens). Uses the openai client
    pointed at the role's BASE_URL/MODEL/API_KEY. If usage is missing from the
    response, estimate tokens as len(text)//4."""
    _, model, _ = ROLES[role]
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client(role).chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    if usage and usage.prompt_tokens is not None:
        p_tok, c_tok = usage.prompt_tokens, usage.completion_tokens or 0
    else:
        p_tok = sum(len(m.get("content", "")) for m in messages) // 4
        c_tok = len(text) // 4
    return text, p_tok, c_tok


def parse_json(text: str) -> dict:
    """Parse model JSON output; on failure strip code fences and retry once."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        return json.loads(stripped)
