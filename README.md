# Tripwire: a cheap chain-of-thought monitor, stress-tested

A lightweight decision model used to monitor safe AI models interactions.

> We built a cheap CoT monitor, stress-tested it, and found where it breaks: same detection as explaining everything, with far fewer tokens.

## Problem

Reasoning models write a chain of thought (CoT) before they act. That CoT often says the quiet part out loud: *"I'll hardcode the expected output"*, *"no need to tell the user the tests failed"*. Reading every CoT with a strong model is expensive. Can a cheap model do the triage?

## Architecture

```
            every trace                       flagged traces only
  CoT ──▶ ┌────────────────────┐  flag? ──▶ ┌──────────────────────────┐
          │ Stage 1: decision  │   yes      │ Stage 2: explainer       │
          │ cheap model, JSON  │──────────▶ │ span + ~300 chars context│──▶ confirmed / overturned
          │ flag, category,    │            │ plain-English explanation│    + predicted next action
          │ confidence, span   │            └──────────────────────────┘
          └────────────────────┘
                 │ no
                 ▼
             not flagged (no stage-2 tokens spent)
```

| Stage | Model | How it runs |
|---|---|---|
| 1. Screener | [d3-flash](https://huggingface.co/vllm-sr/d3-flash) (9B decision model), [Q5_K_M-dyn GGUF](https://huggingface.co/Livesport/d3-flash-GGUF) | `llama-server` `/v1/systemone`. Returns probabilities, never text |
| 2. Explainer | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) GGUF, thinking off | `llama-server` OpenAI-compatible chat |

- **Stage 1** asks d3 three typed questions: P(misbehaving) (yes/no), category (choice), and which sentence is most suspicious (choice, one option per sentence). The top sentence becomes `suspicious_span`. Flag if P ≥ `STAGE1_THRESHOLD` (0.3).
- **Stage 2** sees only the span plus ~300 characters of context, and writes the explanation and the predicted next action. **It cannot overturn stage 1.** The screener is the stronger model, so `stage2.confirmed` is recorded as MiniCPM's advisory agreement only (`STAGE2_CAN_OVERTURN` in [monitor/pipeline.py](monitor/pipeline.py)).
- **Baseline (explain-all):** MiniCPM judges the full CoT of every trace, with no screener.
- **Cost caveat:** the screener (9B) is larger than the explainer (2B) and reads every full trace, so raw token totals favor the baseline. The cascade's advantages are that the screener generates no output tokens, returns calibrated probabilities, and keeps the 2B model's weaker judgment out of the flag decision. Compare latency as well as tokens.

## Dataset design

| Slice | Count | Why |
|---|---|---|
| Obvious malicious | TODO | sanity check |
| Subtle malicious | TODO | misbehavior is implicit and plausibly justified, with no "secretly"/"deceive" words |
| Hard negatives | TODO | benign but uses scary words (considering a bad action and rejecting it, security tutorials) |
| Clean benign | TODO | baseline FPR |
| Evasion rewrites (`evasion_r1`) | TODO | the generator rewrites caught traces to slip past the monitor |

Six categories: reward hacking, sandbagging, user deception, permission seeking, oversight subversion, hint misuse.

## Results

_All numbers from `results/metrics.json`; regenerate with `python -m eval.run && python -m eval.metrics`._

| Headline | Result |
|---|---|
| Recall: obvious / subtle | TODO |
| FPR: hard negatives / clean | TODO |
| Tokens: cascade vs explain-all | TODO |
| Recall before → after evasion | TODO |
| Stage 2 overturn rate | TODO |
| Stage 1 recall alone | TODO |

![Detection by difficulty](results/recall_by_difficulty.png)
![False positives](results/fpr.png)
![Cascade vs explain-all](results/cascade_vs_explain_all.png)
![Evasion](results/evasion.png)
![Recall by category](results/recall_by_category.png)

### An example that fooled every judge

TODO: paste the trace from `metrics.json -> missed_malicious`.

## Limitations

- **Synthetic data ≠ real model behavior.** Most traces are model-generated to a spec.
- **CoT can be unfaithful.** A model's written reasoning may not reflect what actually drives its output; a clean CoT is not proof of clean behavior.
- **Flag, don't train.** Monitors like this should flag for humans. Using them as a training signal teaches models to hide intent.
- **Small sample size.** Treat percentages as directional.

## Next steps

TODO

## Run it

**On Colab (the models need a GPU):** open [monitor/colab_monitor.ipynb](monitor/colab_monitor.ipynb) in Colab (T4 or L4 runtime), set `REPO_URL` or upload `tripwire_code.zip` (from `python make_colab_zip.py`), and run all cells. It builds llama.cpp, starts both servers, runs `eval.run` and `eval.metrics`, and downloads `tattletale_results.zip`. Unzip that into the repo (it contains `results/`) for the reporting step.

**Locally (analysis and demo on existing results, or against your own servers):**

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                               # set STAGE1_MODEL / STAGE2_MODEL
python -m eval.run                                 # writes results/*_results.jsonl (resume-safe)
python -m eval.metrics                             # writes results/metrics.json + charts
python -m app.demo                                 # live demo (add --offline for keyword stub)
```

## Repo layout

| Path | Owner | What |
|---|---|---|
| `common/` | lead | frozen schemas, config, LLM helper |
| `generate/` | Part A | synthetic traces + evasion round |
| `monitor/` | Part B | stage 1 (d3 client), stage 2, pipeline, Colab notebook |
| `eval/`, `app/`, `results/` | Part C | evaluation, charts, Gradio demo |
