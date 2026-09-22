# Qwen3.5-4B Q4 Integration & Calibration Plan

This runbook defines the isolated experimentation and recalibration protocol for evaluating **Qwen 3.5 4B (Q4_K_M)** within OpenSourceJev on consumer hardware (NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM), while keeping the established **Qwen3-1.7B-Q8_0** baseline completely untouched and reproducible.

---

## 1. Executive Summary & Objective

The engineering architecture of OpenSourceJev (native `llama.cpp` logits interception, bounded candidate projection, Kraft-McMillan prefix-free termination, indexed scoring, and Jev wire protocol `/v1/systemone`) transfers directly to any causal language model. However, **calibration values, tokenization boundaries, and prompt whitespace dynamics do NOT transfer across model families or quantization types**.

### Core Questions This Experiment Answers:
1. **CUDA & Runtime Compatibility**: Can Qwen 3.5 4B run through our native `llama.cpp` CUDA dynamic library on Windows?
2. **Prompt & Token Boundary Alignment**: Does Qwen 3.5's ChatML template produce immediate answer-token logits without conversational or reasoning drift?
3. **Reasoning & Distractor Immunity**: Does the jump from 1.7B to 4B parameters resolve lexical distractor traps (e.g. `original-intent-05-0` where mentioning "refund policy" distracted 1.7B from "other")?
4. **Calibration Quality**: What is the optimal temperature scaling $T$ on BoolQ/Noul, and how does Expected Calibration Error (ECE) compare?
5. **VRAM Envelope & Usability**: Can Qwen 3.5 4B Q4_K_M run 100% offloaded to the 4GB RTX 3050 without CPU swapping or out-of-memory errors?
6. **Cost-to-Latency Trade-off**: Does the accuracy gain justify the latency increase over 1.7B?

---

## 2. Baseline Status Freeze (Qwen3-1.7B-Q8)

Before executing new experiments, the baseline is frozen:

* **Branch**: `experiment/qwen35-4b` (branched from `dev` at commit `8ce9c31`)
* **Baseline Commit**: `8ce9c31dbe8c8cdd9556e5b311eecd3f557442e5`
* **Baseline Model**: `models/Qwen3-1.7B-Q8_0.gguf` (1.71 GB)
* **Baseline SHA-256**: `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
* **Baseline Calibration**: `jev_calibration.json` ($T = 9.470457$)
* **Hardware Profile**:
  * GPU: NVIDIA GeForce RTX 3050 Laptop GPU (Compute Capability 8.6, 4095 MiB VRAM)
  * CUDA Driver: 555.99 / CUDA 12.5 (Runtime: CUDA 12.4 via `ggml-cuda.dll`)
  * Runtime: `llama.cpp` 0.4.1-dev (Build 10964) with Flash Attention & CUDA Graphs
* **Verified Benchmark Scorecard**:
  * `original.jsonl` (72 tasks): **69.44% Accuracy** (50/72), ECE 0.164, p50 latency 0.312s
  * `easy.jsonl` (48 tasks): **95.83% Accuracy** (46/48), ECE 0.078, p50 latency 0.705s
  * `hard.jsonl` (111 tasks): **39.64% Accuracy** (44/111), ECE 0.453, p50 latency 0.922s
  * Strict Schema Validity: **100.0%** across all 231 decisions

---

## 3. Phase-by-Phase Experimental Protocol

### Phase 1: Branch & Directory Isolation
- All work occurs on `experiment/qwen35-4b`.
- Model files live in an isolated directory: `models/qwen35-4b-q4km/`.
- Calibration output lives in `jev_calibration_qwen35_4b_q4km.json`.
- The original `models/Qwen3-1.7B-Q8_0.gguf` and `jev_calibration.json` remain completely unmodified.

### Phase 2: Model Ingestion & Verification
- **Target File**: `Qwen3.5-4B-Q4_K_M.gguf` (~2.55 GB) from `unsloth/Qwen3.5-4B-GGUF` (Revision pinned).
- Download using `curl.exe` with byte-range resume support into `models/qwen35-4b-q4km/`.
- Compute and verify SHA-256 hash against Hugging Face metadata.
- Verify that disk space remains $> 40\text{ GB}$ on `C:`.

### Phase 3: Independent Runtime Smoke Test & VRAM Profiling
Before integrating into the web app, test the model in isolation with `llama-cli.exe`:
```powershell
.\runtime\llama.cpp\bin\llama-cli.exe `
  -m .\models\qwen35-4b-q4km\Qwen3.5-4B-Q4_K_M.gguf `
  -ngl 99 `
  -c 2048 `
  -p "<|im_start|>user`nClassify: The payment failed.`nAllowed: billing, technical`n<|im_end|>`n<|im_start|>assistant`n<think>`n`n</think>`n`nANSWER:" `
  -n 4
```
**Verification Gates**:
1. Full GPU offload succeeds without CUDA OOM (record `nvidia-smi` peak VRAM).
2. KV-cache allocation with $n_{\text{ctx}} = 2048$ leaves at least $600\text{ MB}$ free VRAM.
3. Model does not require vision projectors or external multimodal weights for text inference.

### Phase 4: Tokenizer, Whitespace, & Prefix Alignment Audit
Audit Qwen 3.5's BPE tokenization to ensure candidates are single, aligned tokens:
1. **Noul Tokens**: Test `" true"` vs `"true"` and `" false"` vs `"false"`.
2. **Indexed Choice Tokens**: Test `" A"`, `" B"`, `" C"`, `" D"`.
3. **Score Index Tokens**: Test `" 0"`, `" 1"`, `" 2"`, `" 3"`, `" 4"`, `" 5"`.
4. **Multi-Token Stem / Prefix Collisions**: Verify whether `" coding"` vs `" coding_agent"` behaves identically or whether delimiter termination is required.
5. **Turn Alignment**: Confirm whether `<|im_start|>assistant\n<think>\n\n</think>\n\nANSWER:` yields the maximum next-token probability density for answer candidates, or if Qwen 3.5 expects an alternate thinking suppress syntax.

### Phase 5: Model-Scoped Architecture & Engine Support
Update `app/native_engine.py` and `app/main.py` to allow multi-model configuration without hardcoding:
- Support `JEV_LLAMA_MODEL` pointing to `models/qwen35-4b-q4km/Qwen3.5-4B-Q4_K_M.gguf`.
- Implement dynamic calibration file loading:
  $$\text{calibration\_path} = \text{resolve\_calibration}(model\_path)$$
  If evaluating Qwen 3.5, load `jev_calibration_qwen35_4b_q4km.json`.
  If evaluating Qwen 3 1.7B, load `jev_calibration.json`.

### Phase 6: Temperature Recalibration on BoolQ
The temperature $T = 9.470457$ was fitted specifically to Qwen3-1.7B-Q8.
For Qwen 3.5 4B Q4:
1. Run unscaled raw logits over the 200-sample BoolQ calibration split.
2. Minimize Negative Log-Likelihood (NLL) using L-BFGS-B / Nelder-Mead:
   $$\min_T -\sum_{i=1}^N \log \sigma\left(\frac{z_{\text{true}, i} - z_{\text{false}, i}}{T}\right)$$
3. Validate on held-out validation split:
   - Measure ECE, Brier Score, and NLL before and after scaling.
   - Save fitted temperature to `jev_calibration_qwen35_4b_q4km.json`.

### Phase 7: Controlled Ablation Matrix
To attribute improvements accurately (separating model capacity gains from inference techniques), execute:
1. **Ablation A**: Qwen 3.5 Raw Logits (Uncalibrated, literal strings, no delimiters).
2. **Ablation B**: Qwen 3.5 + Prompt Alignment (`ANSWER:` token boundary).
3. **Ablation C**: Qwen 3.5 + Indexed Score (`" 0"`, `" 1"`, ...).
4. **Ablation D**: Qwen 3.5 + Kraft-McMillan Prefix-Free Delimiters (`\n` on collisions).
5. **Ablation E**: Qwen 3.5 + Refitted Noul Temperature Scaling.

### Phase 8: Full 3-Tier JevBench Benchmark
Run the canonical JevBench test suite against the local server using Qwen 3.5:
- `original.jsonl` (72 tasks)
- `easy.jsonl` (48 tasks)
- `hard.jsonl` (111 tasks)

Compare against Qwen 3-1.7B across:
- Overall Accuracy
- Per-Family Accuracy (Routing, Intent, Extraction, Ordinal, Policy, Adequacy)
- ECE & Brier Calibration Score
- Median (p50) & 95th-percentile (p95) Latency
- VRAM Footprint

---

## 4. Hardware Acceptance Criteria (RTX 3050 4GB)

The Qwen 3.5 4B model will be designated a **Production Success** if and only if:
1. **Zero OOM**: Allocates within the 4096 MiB VRAM boundary with $n_{\text{ctx}} \ge 2048$.
2. **Schema Validity**: 100.0% strict validity maintained across all 231 decisions.
3. **Accuracy Gain**: Improves accuracy on held-out tasks (`hard.jsonl` and/or `original.jsonl`).
4. **Latency Budget**: Median latency $\le 800\text{ms}$ per decision on GPU.
5. **Reproducibility**: Baseline Qwen 3 1.7B can be swapped back in with zero performance degradation.

---

## 5. Required Deliverables
1. Downloaded & hash-verified `Qwen3.5-4B-Q4_K_M.gguf`.
2. Tokenizer & prompt alignment audit log.
3. `jev_calibration_qwen35_4b_q4km.json`.
4. Full JevBench JSON summaries in `benchmark_results/qwen35_*/`.
5. Comparison report in `docs/BENCHMARK_QWEN35_VS_QWEN3.md`.
