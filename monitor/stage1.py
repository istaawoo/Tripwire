"""Stage 1 dispatcher. Env STAGE1_BACKEND picks the decision model:
  d3-cache (default) verdicts precomputed on a GPU by `python -m monitor.d3_cache`; falls back to llm per trace if missing
  d3 / d3-http       live vllm-sr/d3-flash system_one (GPU / Colab, or vllm-sr serve)
  llm                small chat model through common.llm (STAGE1_MODEL, e.g. qwen3:4b)
Final flag = confidence >= STAGE1_THRESHOLD (low on purpose: recall first)."""
import os
import sys

from common import config
from common.schemas import Trace

from . import stage1_llm

THRESHOLD = float(getattr(config, "STAGE1_THRESHOLD", 0.3))
LAST_BACKEND = {"name": None}  # which backend answered the most recent call (for the demo badge)


def classify(trace: Trace):
    """-> (Stage1Result, in_tokens, out_tokens)"""
    backend = os.getenv("STAGE1_BACKEND", "d3-cache")
    if backend.startswith("d3"):
        from . import stage1_d3
        res = stage1_d3.classify(trace, THRESHOLD)
        if res is not None:
            LAST_BACKEND["name"] = "d3-flash"
            return res
        print(f"[stage1] no d3 verdict cached for {trace.id}; using LLM fallback", file=sys.stderr)
    LAST_BACKEND["name"] = f"llm:{config.STAGE1_MODEL}"
    return stage1_llm.classify(trace)
