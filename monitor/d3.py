"""Minimal client for d3-flash served by llama.cpp's `llama-server` on POST /v1/systemone.

d3 is a decision model: it does not generate text, it returns a probability for
every answer to typed questions (noul = yes/no, choice, score) about a `state`.
"""
import json
import urllib.error
import urllib.request

from common.config import STAGE1_BASE_URL

ENDPOINT = STAGE1_BASE_URL.rstrip("/").removesuffix("/v1") + "/v1/systemone"


def systemone(state, questions: dict, timeout=120) -> tuple[dict, int]:
    """Returns (answers, input_tokens). One request; raises on HTTP errors."""
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"d3 HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    usage = data.get("usage") or {}
    return data.get("answers", {}), int(usage.get("input_tokens", 0))


def p_true(answer) -> float:
    """Probability of 'true' for a noul answer, tolerant of response-shape variations."""
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, dict):
        for k in ("noul", "probability", "p", "true"):
            v = answer.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        probs = answer.get("probabilities")
        if isinstance(probs, dict):
            for k in ("true", "yes", True):
                if k in probs:
                    return float(probs[k])
    raise ValueError(f"can't read noul probability from {answer!r}")


def choice_probs(answer) -> dict[str, float]:
    """Option -> probability for a choice answer."""
    if isinstance(answer, dict):
        probs = answer.get("probabilities")
        if isinstance(probs, dict):
            return {k: float(v) for k, v in probs.items()}
        if "choice" in answer:
            return {answer["choice"]: float(answer.get("confidence", 1.0))}
    raise ValueError(f"can't read choice probabilities from {answer!r}")


def ask_noul(state, instructions, criteria=None) -> tuple[float, int]:
    q = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    answers, tok = systemone(state, {"q": q})
    return p_true(answers["q"]), tok


def ask_choice(state, instructions, criteria: dict) -> tuple[dict[str, float], int]:
    answers, tok = systemone(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})
    return choice_probs(answers["q"]), tok
