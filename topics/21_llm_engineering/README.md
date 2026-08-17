# Topic 21: LLM Engineering

> **Phase:** 5 | **Status:** ✅ Complete | **Prerequisites:** [13 Neural Networks](../13_neural_networks/README.md), [16 Transformer](../16_transformer/README.md)

## Overview

The modern LLM pipeline consists of pre-training, parameter-efficient fine-tuning (PEFT), and alignment. This module covers Byte-Pair Encoding (BPE) for subword tokenization, Low-Rank Adaptation (LoRA) and QLoRA for efficient fine-tuning, and Reinforcement Learning from Human Feedback (RLHF) along with Direct Preference Optimization (DPO) for model alignment.

## Scope

- **In scope:** word-level BPE (the classic Sennrich et al. algorithm, matching `theory.md` and the tested package implementation), LoRA rank decomposition and parameter accounting, and the DPO objective in stable log-space — all in pure NumPy. Byte-level BPE is described as a real-world variant.
- **Out of scope:** training or serving actual language models, QLoRA quantization internals, and RLHF reward-model training.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | [`theory.md`](theory.md) | Theory | BPE tokenization algorithm, LoRA derivations and cost analysis, RLHF pipeline, DPO objective derivation, failure modes |
| 2 | [`first_principles.ipynb`](first_principles.ipynb) | Computation | WHY→BUILD→VERIFY — from-scratch NumPy implementations of BPE Tokenizer, LoRA Linear layer, and DPO loss |
| 3 | [`exercises.ipynb`](exercises.ipynb) | Practice | Hand calculation of BPE merges, LoRA parameter savings computation, conceptual analysis of DPO vs RLHF |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/llm_models.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/llm_models.py) (`BPETokenizer` with `fit`/`encode`/`decode`, `LoRALinear`, `dpo_loss`), covered by `tests/test_phase5_models.py`.

## Connections

- **Prereqs:** [13 Neural Networks](../13_neural_networks/README.md), [16 Transformer](../16_transformer/README.md)
- **Synthesis:** Connections to [Optimization Methods Compared](../../synthesis/optimization_methods_compared.md)
- **Next:** Advanced inference optimization (KV caching, speculative decoding, continuous batching)
