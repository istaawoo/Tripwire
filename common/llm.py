"""One chat helper for every role. FROZEN after initial commit."""
import json
import re
from functools import lru_cache
from typing import Literal

from openai import OpenAI

from .config import ROLES


@lru_cache(maxsize=None)
def _client(base_url, api_key):
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300)


def _strip_fences(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()


def chat(role: Literal["generator", "stage1", "stage2"], messages, json_mode=False, max_tokens=512) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens). Uses the openai client
    pointed at the role's BASE_URL/MODEL/API_KEY. If usage is missing from the
    response, estimate tokens as len(text)//4. In json_mode the returned text is
    valid JSON when the model produced any (code fences stripped)."""
    base_url, model, key = ROLES[role]
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _client(base_url, key).chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0, **kw)
    text = resp.choices[0].message.content or ""
    if json_mode:
        try:
            json.loads(text)
        except Exception:
            cleaned = _strip_fences(text)
            try:
                json.loads(cleaned)
                text = cleaned
            except Exception:
                pass  # caller handles unparseable JSON
    u = getattr(resp, "usage", None)
    pin = getattr(u, "prompt_tokens", None) or sum(len(str(m.get("content", ""))) for m in messages) // 4
    pout = getattr(u, "completion_tokens", None) or len(text) // 4
    return text, int(pin), int(pout)
