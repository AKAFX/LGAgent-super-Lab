<div align="center">

# LGAgent

### A Structured and Rule-Guided Multi-Agent Framework for Interpretable Legal Reasoning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Training--Free-7C3AED)](#method)
[![License](https://img.shields.io/badge/License-Apache--2.0-2EA44F)](LICENSE.txt)
[![Paper](https://img.shields.io/badge/Paper-Under%20Review-B31B1B)](#citation)

**Structured decomposition · Process control · Evidence-aware verification · Bounded interaction**

[Overview](#overview) · [Method](#method) · [Results](#results) · [Quick Start](#quick-start) · [Evaluation](#evaluation)

</div>

---

## Overview

Large language models are capable legal reasoners, but a single end-to-end model often mixes issue identification, rule application, evidence checking, and final judgment inside an opaque reasoning process. **LGAgent** makes those responsibilities explicit.

LGAgent is a training-free, rule-guided multi-agent framework that coordinates three specialized roles:

- **Lawyer A - Structured Case Analyst:** decomposes a legal question into option-level claims, legal keywords, and potential traps without expressing an answer preference.
- **Judge - Process Controller:** produces auditable control signals, evidence requirements, counterexample directions, and an optional retrieval decision.
- **Lawyer B - Decision Maker:** first gives a blind answer with confidence, then performs evidence-aware option verification and bounded clarification before returning the final answer.

Across JEC-QA, LexGenius, and CAIL2022, the framework consistently improves a wide range of backbones without fine-tuning.

<div align="center">
  <img src="docs/assets/lgagent-overview.png" alt="LGAgent overview" width="96%">
</div>

## Method

<div align="center">
  <img src="docs/assets/lgagent-framework.png" alt="LGAgent framework" width="96%">
</div>

The reasoning pipeline is deliberately constrained:

1. **Structure first.** Lawyer A converts the question and options into an answer-neutral JSON scaffold.
2. **Control explicitly.** The Judge determines evidence needs, possible counterexamples, and whether retrieval should be activated.
3. **Separate prior belief from verification.** Lawyer B produces a blind prediction (`B0`) before seeing the upstream structure, then verifies every option in `B1` using `SUPPORT`, `REFUTE`, or `NEI`.
4. **Continue only under uncertainty.** Clarification is triggered when the proportion of `NEI` labels reaches `0.5` or blind confidence falls below `0.6`, with at most two additional rounds.
5. **Keep retrieval optional.** The retrieval branch is implemented and Judge-controlled; the paper's main experiments set `USE_RAG=False` to isolate gains from the collaboration protocol.

## Results

### Main Results

Exact Match (EM) on three Chinese legal reasoning benchmarks. Each cell reports `Direct / CoT / LGAgent`.

| Backbone | JEC-QA | LexGenius | CAIL2022 |
|:--|:--:|:--:|:--:|
| DeepSeek-V3 | 61.65 / 64.75 / **74.00** | 62.94 / 63.17 / **69.90** | 65.37 / 64.55 / **72.75** |
| DeepSeek-R1 | 72.56 / 73.42 / **81.20** | 66.29 / 69.58 / **86.97** | 79.45 / 81.85 / **89.72** |
| Qwen3-4B | 43.03 / 44.15 / **57.33** | 55.62 / 56.33 / **59.25** | 59.42 / 61.28 / **70.63** |
| GLM-4-Air | 69.60 / 70.00 / **77.95** | 55.28 / 56.13 / **62.47** | 64.57 / 64.54 / **72.00** |
| GPT-4o mini | 40.56 / 41.25 / **51.40** | 55.71 / 54.12 / **59.00** | 58.33 / 56.46 / **67.84** |

The gains are not concentrated in a single category. On LexGenius, the multi-agent protocol improves legal understanding, reasoning, application, ethics, language, law and society, and judicial practice.

<div align="center">
  <img src="docs/assets/domain-gains.png" alt="Domain-level gains" width="94%">
</div>

### Ablation Study

| Setting | JEC-QA | LexGenius | CAIL2022 |
|:--|--:|--:|--:|
| Single model | 61.65 | 62.94 | 65.37 |
| Without structured parsing | 70.23 | 67.67 | 70.65 |
| Without Judge reasoning | 70.11 | 66.42 | 70.68 |
| Single-turn verification | 71.35 | 66.94 | 71.83 |
| Multi-turn verification | 71.47 | 67.12 | 71.54 |
| **Full LGAgent** | **74.00** | **69.90** | **72.75** |

### Cost-Performance Trade-off

LGAgent turns economical backbones into competitive legal reasoners. In the sampled cost analysis, LGAgent configurations cost **$0.29-$0.68 per 1,000 questions**, compared with **$3.16-$3.37** for the evaluated frontier direct-answer baselines.

<div align="center">
  <img src="docs/assets/cost-performance.png" alt="Performance and inference cost" width="90%">
</div>

## Quick Start

### 1. Create the environment

```bash
./scripts/bootstrap_git_lfs.sh
uv sync --python 3.11 --no-dev
source .venv/bin/activate
```

The bootstrap script installs the pinned Git LFS client under `.tools/`, configures
this repository only, and downloads the dataset objects. Verify the reproducible
baseline without credentials or network access:

```bash
python tools/baseline_check.py
python -m unittest tests.test_baseline_check
```

### 2. Configure an OpenAI-compatible model

```bash
cp .env.example .env
```

Set `LLM_API_KEY` in `.env`, then update the model name and `base_url` in:

```text
examples/parameter/legal2_rag_parameter.yaml
```

When `api_key` is set in that YAML file it takes precedence; `LLM_API_KEY` is
used only when the YAML value is empty. Run the credentialed single-question
smoke test explicitly:

```bash
python tools/baseline_check.py --online-smoke
```

For concurrent evaluation, optional comma-separated key pools can be supplied through:

```text
LGAGENT_REASONING_API_KEYS
LGAGENT_EVAL_API_KEYS
```

### 3. Run a single legal question

```bash
python tools/legal_multi_agent_prompt_demo.py \
  "A complete legal multiple-choice question with options A, B, C, and D."
```

## Evaluation

Run the full configurable evaluator:

```bash
python tools/legal_multi_agent_eval_optimized.py \
  --dataset data/Ability_sample_100.jsonl \
  --eval-model deepseek-v3 \
  --enable-lawyer-a true \
  --enable-judge true \
  --enable-dialogue true \
  --use-rag false \
  --concurrency 8
```

Run the automated ablation suite:

```bash
python tools/legal_multi_agent_ablation_auto.py \
  --datasets data/Ability_sample_100.jsonl \
  --eval-model deepseek-v3 \
  --concurrency 8
```

Evaluation outputs and checkpoints are written under `output/`, which is excluded from version control.

## Repository Structure

```text
LGAgent/
├── docs/assets/                     # Paper figures used by this README
├── examples/parameter/              # Reproducible pipeline configuration
├── prompt/                          # Legal QA prompt templates
├── servers/                         # Modular MCP services inherited from UltraRAG
├── src/ultrarag/                    # Pipeline runtime and client
├── tools/
│   ├── legal_multi_agent_prompt_demo.py
│   ├── legal_multi_agent_eval_optimized.py
│   ├── legal_multi_agent_ablation_auto.py
│   └── baseline_eval.py
└── data/                            # Small examples only; full datasets are external
```

## Design Principles

- **Auditable role boundaries:** every agent has a narrow, inspectable responsibility.
- **Structured intermediate state:** coordination uses JSON-like signals instead of unrestricted conversation.
- **Deterministic process control:** clarification and retrieval are activated by explicit rules.
- **Backbone agnostic:** any OpenAI-compatible model can serve as a reasoning backend.
- **Reproducible experimentation:** checkpointing, concurrent execution, baselines, and automated ablations are included.

## Citation

The paper is currently under review. Citation metadata will be updated when a public preprint is available.

```bibtex
@misc{feng2026lgagent,
  title  = {LGAgent: A Structured and Rule-Guided Multi-Agent Framework for Interpretable Legal Reasoning},
  author = {Xin Feng and Haoming Liu and Manjia Feng and Feng Shu and Yi Feng},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgements

LGAgent is implemented on top of [UltraRAG 2.0](https://github.com/OpenBMB/UltraRAG), whose modular MCP-based pipeline provides the underlying retrieval, generation, evaluation, and orchestration infrastructure.

## License

This repository is released under the [Apache License 2.0](LICENSE.txt).
