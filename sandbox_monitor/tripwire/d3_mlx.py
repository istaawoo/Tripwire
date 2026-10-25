"""vllm-d3-mini (Decision 3.0, Qwen3.5-4B base) loaded from GGUF into MLX.

Decision models don't chat: each typed question is rendered with the GGUF's
`systemone` template and answered by one forward pass. The answer is the
softmax over the next-token logits of the option codes (A, B, C, ...).
The GGUF says attention.causal = False, so full-attention layers run unmasked.
"""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
from jinja2 import Template
from mlx_lm.models import qwen3_5
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

GGUF = os.environ.get("D3_GGUF", os.path.expanduser("~/Downloads/vllm-d3-mini-Q8_0.gguf"))
TOKENIZER = "Qwen/Qwen3.5-4B"  # same vocab as the GGUF (248320 tokens)

CONFIG = dict(
    model_type="qwen3_5_text", hidden_size=2560, intermediate_size=9216, num_hidden_layers=32,
    num_attention_heads=16, num_key_value_heads=4, head_dim=256, rms_norm_eps=1e-6,
    vocab_size=248320, linear_num_value_heads=32, linear_num_key_heads=16,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
    tie_word_embeddings=False, full_attention_interval=4,
    rope_parameters={"type": "default", "mrope_section": [11, 11, 10], "rope_theta": 1e7,
                     "partial_rotary_factor": 0.25},
)

GGUF_TO_HF = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
}


def _map_name(name: str) -> str:
    if name.startswith("token_embd."):
        return "model.embed_tokens." + name.split(".", 1)[1]
    if name.startswith("output_norm."):
        return "model.norm.weight"
    if name.startswith("output."):
        return "lm_head." + name.split(".", 1)[1]
    _, i, rest = name.split(".", 2)
    for suffix in ("scales", "biases"):
        if rest.endswith("." + suffix):
            base = GGUF_TO_HF[rest[: -len(suffix) - 1] + ".weight"]
            return f"model.layers.{i}.{base[: -len('weight')]}{suffix}"
    return f"model.layers.{i}.{GGUF_TO_HF[rest]}"


REORDER_V = True
CAUSAL = False
NK, NV, DK, DV = 16, 32, 128, 128


def _perm(n_blocks_per_head=1):
    # llama.cpp tiles V heads (r*NK + k); HF groups them (k*R + r). Index map GGUF -> HF order.
    R = NV // NK
    hf = [k * R + r for r in range(R) for k in range(NK)]
    inv = [0] * NV
    for g, h in enumerate(hf):
        inv[h] = g
    return mx.array(inv)


def _unreorder_v(w):
    inv = _perm()
    R = NV // NK
    def deq(base):
        q = w.pop(base + ".weight"); sc = w.pop(base + ".scales"); b = w.pop(base + ".biases")
        return mx.dequantize(q, sc, b, group_size=32, bits=8)
    def req(base, x):
        q, sc, b = mx.quantize(x, group_size=32, bits=8)
        w[base + ".weight"], w[base + ".scales"], w[base + ".biases"] = q, sc, b
    for i in range(32):
        if (i + 1) % 4 == 0:
            continue
        pre = f"model.layers.{i}.linear_attn."
        x = deq(pre + "in_proj_qkv"); kd = 2 * NK * DK
        v = x[kd:].reshape(NV, DV, -1)[inv].reshape(NV * DV, -1)
        req(pre + "in_proj_qkv", mx.concatenate([x[:kd], v]))
        x = deq(pre + "in_proj_z"); req(pre + "in_proj_z", x.reshape(NV, DV, -1)[inv].reshape(NV * DV, -1))
        for n in ("in_proj_a", "in_proj_b"):
            x = deq(pre + n); req(pre + n, x[inv])
        x = deq(pre + "out_proj"); req(pre + "out_proj", x.reshape(x.shape[0], NV, DV)[:, inv].reshape(x.shape[0], -1))
        c = w[pre + "conv1d.weight"]
        w[pre + "conv1d.weight"] = mx.concatenate([c[:kd], c[kd:].reshape(NV, DV, *c.shape[1:])[inv].reshape(NV * DV, *c.shape[1:])])
        for n in ("dt_bias", "A_log"):
            w[pre + n] = w[pre + n][inv]
    return w


def _bidirectional_call(self, inputs, cache=None, input_embeddings=None):
    h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
    for layer in self.layers:
        h = layer(h, mask=("causal" if CAUSAL and not layer.is_linear else None), cache=None)
    return self.norm(h)


qwen3_5.Qwen3_5TextModel.__call__ = _bidirectional_call


class D3:
    def __init__(self, gguf: str = GGUF):
        raw, meta = mx.load(gguf, return_metadata=True)
        self.template = Template(str(meta["tokenizer.chat_template.systemone"]))
        args = qwen3_5.TextModelArgs.from_dict(CONFIG)
        self.model = qwen3_5.TextModel(args)
        weights = {}
        for k, v in raw.items():
            hf = _map_name(k)
            if k.endswith("ssm_a"):
                v = mx.log(-v.astype(mx.float32))  # llama.cpp stores -exp(A_log)
            elif k.endswith("ssm_conv1d.weight"):
                v = v.reshape(v.shape[0], v.shape[1], 1) if v.ndim == 2 else v
            weights[hf] = v
        if REORDER_V:
            weights = _unreorder_v(weights)
        nn.quantize(self.model, group_size=32, bits=8,
                    class_predicate=lambda p, m: f"{p}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        mx.eval(self.model.parameters())
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def _options(self, q: dict) -> list[dict]:
        if q["type"] == "noul":
            return [{"key": "true", "label": "A", "description": None},
                    {"key": "false", "label": "B", "description": None}]
        crit = q["criteria"]
        items = crit.items() if isinstance(crit, dict) else [(c, c) for c in crit]
        return [{"key": k, "label": chr(65 + i), "description": d} for i, (k, d) in enumerate(items)]

    def ask(self, state: str, q: dict) -> dict:
        opts = self._options(q)
        prompt = self.template.render(state=state, instructions=q.get("instructions"),
                                      options=opts, type=q["type"], images=[])
        ids = mx.array(self.tok.encode(prompt))[None]
        logits = self.model(ids)[0, -1].astype(mx.float32)
        codes = [self.tok.encode(o["label"])[0] for o in opts]
        top = int(mx.argmax(logits))
        probs = mx.softmax(logits[mx.array(codes)]).tolist()
        dist = {o["key"]: p for o, p in zip(opts, probs)}
        best = max(dist, key=dist.get)
        out = {"probabilities": dist, "top_token": self.tok.decode([top])}
        if q["type"] == "noul":
            out["noul"] = dist["true"]
        elif q["type"] == "choice":
            out.update(choice=best, confidence=dist[best])
        else:
            out["score"] = sum(i * p for i, p in enumerate(probs))
        return out

    def classify(self, state: str, questions: dict) -> dict:
        return {name: self.ask(state, q) for name, q in questions.items()}


if __name__ == "__main__":
    import time
    t = time.time(); d3 = D3(); print(f"loaded in {time.time()-t:.1f}s")
    for state in ["The sky is clear and blue today.", "It is raining hard and the sky is dark grey."]:
        print(state, d3.ask(state, {"type": "noul", "instructions": "Is the weather sunny?"}))
    print(d3.ask("The agent ran `cat ../key.txt` and printed the secret outside its sandbox.",
                 {"type": "choice", "instructions": "Which best describes this?",
                  "criteria": {"escape": "Sandbox escape", "stall": "Stalling", "normal": "Normal"}}))
