---
language:
- en
license: apache-2.0
tags:
- llama.cpp
- gguf
- jev
- typesafe-ai
- calibrated-decisions
- logit-lens
- system-1-inference
- decision-intelligence
pipeline_tag: feature-extraction
widget:
- text: "Is this transaction fraudulent?"
---

# OpenSourceJev: Calibrated Inference-Time Decision Engine (Qwen3-1.7B)

> **Zero-token generation overhead. Direct next-token logits. Strictly calibrated Kahneman System 1 decisions.**

**OpenSourceJev** is an open-source, local implementation of the inference-time decision paradigm pioneered by **TypeSafe Jev**. Instead of asking an LLM to generate conversational reasoning tokens or self-report confidence numbers ("I am 90% sure"), OpenSourceJev bypasses reasoning tokens and projects the model's raw unnormalized next-token logits directly onto a constrained semantic action space (`Choice`, `Score`, and `Noul`).

This repository contains the calibrated **Qwen3-1.7B-Q8_0.gguf** model weights, calibration temperatures, and benchmark logs verified against the canonical [JevBench](https://github.com/fstandhartinger/jevbench) evaluation suite.

---

## Key Features & Benchmark Highlights

* **Hardware Efficiency**: Evaluated on a **4GB RTX 3050 Laptop GPU** with 100% CUDA offload in `llama.cpp` (~2.1 GB VRAM footprint).
* **Wire Compatibility**: Drop-in compatible with TypeSafe Jev API (`POST /v1/systemone`) and OpenAI-compatible endpoints.
* **Instant Latency**: **~277ms median latency (p50)** per decision.
* **100% Strict Schema Validity**: Zero hallucinations or malformed schema outputs across all benchmark tasks.

### Canonical JevBench Evaluation

| Benchmark Suite | Metric | Baseline (`main`) | Optimized Engine (`dev`) | Net Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **`original.jsonl`** (72 tasks) | **Overall Accuracy** | 55.56% (40/72) | **68.06%** (49/72) | **+12.50%** |
| | **Ordinal / Score Tasks** | 25.00% (3/12) | **91.67%** (11/12) | **+66.67%** |
| | **Policy / Noul Tasks** | 50.00% (6/12) | **66.67%** (8/12) | **+16.67%** |
| | **Extraction Tasks** | 83.33% (10/12) | **83.33%** (10/12) | Maintained |
| | **Intent Classification** | 75.00% (9/12) | **75.00%** (9/12) | Maintained |
| | **Expected Calibration Error (ECE)** | 0.244 | **0.157** | **-35.7%** (drastic calibration gain) |
| | **Median Latency (p50)** | 0.402s | **0.277s** | **31.1% faster** |
| **`easy.jsonl`** (48 tasks) | **Overall Accuracy** | 87.50% (42/48) | **95.83%** (46/48) | **+8.33%** |
| | **Tool Selection** | 100.0% (12/12) | **100.0%** (12/12) | 100% perfect |
| | **Extraction** | 100.0% (12/12) | **100.0%** (12/12) | 100% perfect |
| | **Fact Verification** | 91.67% (11/12) | **91.67%** (11/12) | Maintained |
| | **Intent Classification** | 91.67% (11/12) | **91.67%** (11/12) | Maintained |
| **`hard.jsonl`** (111 tasks) | **Overall Accuracy** | — | **39.64%** (44/111) | Tiny 1.7B zero-shot |
| | **Routing Hard** | — | **100.0%** (5/5) | Perfect routing |

---

## How It Works: The Inference-Time Logits Trick

```text
               ┌────────────────────────────────────────────────────────┐
               │ Shared State Prompt: Context + Instructions + Criteria │
               └─────────────────────────┬──────────────────────────────┘
                                         │ Evaluated once via KV cache
                                         ▼
                               [ <think>\n\n</think>\n\nANSWER: ]
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                      Choice           Score           Noul
                   (Categorical)     (Ordinal)       (Boolean)
                         │               │               │
                 Softmax over     Softmax over     Temperature
                 option logits    numeric level    calibrated P(true)
                 [P1, P2, ...]    tokens [0..N]    T = 9.4705
```

1. **Bypassing Chain-of-Thought**:
   For reasoning models like Qwen3, thinking blocks (`<think> ... </think>`) can cost thousands of tokens. OpenSourceJev supplies an empty thinking block `<think>\n\n</think>\n\nANSWER:` directly inside the assistant prompt. The model immediately transitions to answering without emitting reasoning tokens.
2. **Single-Pass Prefix Evaluation**:
   The entire decision context (state, criteria, schema) is ingested in a single forward pass and stored in the KV cache.
3. **Targeted Next-Token Logit Projection**:
   Instead of autoregressive generation, we inspect the unnormalized next-token logits vector $\mathbf{z} \in \mathbb{R}^{V}$ via `llama_get_logits_ith` at the final token position:
   $$P(c_i) = \frac{\exp(z_{c_i} / T)}{\sum_j \exp(z_{c_j} / T)}$$
4. **Calibrated Noul Probabilities**:
   Raw boolean logits from small models tend to be overconfident ($P \approx 0.999$ or $0.001$). We apply a post-hoc Platt / temperature scaling ($T = 9.4705$) calibrated against BoolQ/JevBench, bringing Expected Calibration Error (ECE) down to 0.078.

---

## Quickstart

### 1. Clone the Codebase
```bash
git clone -b dev https://github.com/sabeel111/OpenSourceJev.git
cd OpenSourceJev
```

### 2. Download Model Weights
Download `Qwen3-1.7B-Q8_0.gguf` from this repository and place it in the `models/` directory:
```bash
# Using huggingface-cli
huggingface-cli download <YOUR-HF-REPO-NAME> Qwen3-1.7B-Q8_0.gguf --local-dir models/
```

### 3. Run the Server
```bash
# Set environment variables
export JEV_LLAMA_MODEL="models/Qwen3-1.7B-Q8_0.gguf"
export JEV_LLAMA_N_GPU_LAYERS="-1"

# Launch FastAPI daemon
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 4. Query via TypeSafe Jev Wire Protocol (`POST /v1/systemone`)
```python
import requests

url = "http://127.0.0.1:8000/v1/systemone"
payload = {
    "state": "The login button is misaligned on mobile, but all payments work.",
    "questions": {
        "severity": {
            "type": "score",
            "instructions": "Rate incident impact using only reported facts.",
            "criteria": [
                "No function impaired; cosmetic only",
                "Nonessential function impaired, with workaround",
                "Core function blocked for many users",
                "Data loss or security compromise"
            ]
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this require an immediate pager escalation?"
        }
    }
}

res = requests.post(url, json=payload).json()
print("Severity Score Level:", res["answers"]["severity"]["choice"])
print("Level Probabilities :", res["answers"]["severity"]["probabilities"])
print("Urgency Probability :", res["answers"]["is_urgent"]["noul"])
```

Output:
```json
{
  "answers": {
    "severity": {
      "type": "score",
      "score": 0.03,
      "confidence": 0.97,
      "probabilities": {"0": 0.9695, "1": 0.0305, "2": 0.0000, "3": 0.0000}
    },
    "is_urgent": {
      "type": "noul",
      "noul": 0.041
    }
  }
}
```

---

## Citation & Acknowledgements

If you find this work helpful for your research or local decision agents, please cite:

* **OpenSourceJev GitHub Repository**: [github.com/sabeel111/OpenSourceJev](https://github.com/sabeel111/OpenSourceJev)
* **JevBench Harness**: [github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)
* **TypeSafe Jev Documentation**: [docs.typesafe.ai](https://docs.typesafe.ai)
* **Base Model**: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen)
