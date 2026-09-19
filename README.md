# OpenSourceJev ⚡

> **A high-performance, local "System One" AI decision engine powered by `llama.cpp` and Qwen.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![llama.cpp](https://img.shields.io/badge/backend-llama.cpp%20CUDA-orange.svg)](https://github.com/ggerganov/llama.cpp)
[![ViZDoom](https://img.shields.io/badge/demo-ViZDoom%20Agent-red.svg)](https://vizdoom.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OpenSourceJev** is an open-source, local implementation of the **"System One" AI decision paradigm** popularized by TypeSafe AI's Jev. Instead of using Large Language Models (LLMs) to generate verbose text or stream fragile JSON blobs that require retry loops and regex parsing, OpenSourceJev turns open-weight models (like **Qwen3-1.7B**) into ultra-fast, deterministic, strictly-typed decision engines.

It runs locally on consumer laptops (tested on an RTX 3050 Laptop GPU with 4GB VRAM) delivering sub-100ms structured judgments.

---

## 💡 Why "System One"?

Traditional frontier LLMs (GPT-4o, Claude, o1) represent **"System 2"** thinking: slow, deliberative, autoregressive, token-by-token generation. 

Software architectures and autonomous agents often don't need conversational prose. They need **fast, reliable, probabilistic branching** ("smart `if`-statements"):
* *"Which department handles this support ticket?"*
* *"Does this message violate our safety policy?"*
* *"What tactical action should the agent execute right now?"*

```
Traditional LLM Approach:
[ State ] ──> [ Heavy LLM ] ──> [ Generated JSON string ] ──> [ Regex / Pydantic Parser ] ──> [ Error / Retry? ]

OpenSourceJev Approach:
[ State ] ──> [ Shared Prefix ] ──> [ Native Next-Token Logits ] ──> [ Candidate-Only Softmax ] ──> [ Typed Decision ]
                                                                       (Zero JSON Syntax Errors)
```

---

## 🚀 Key Features

* **Guaranteed Type Safety (Zero Syntax Errors)**: By projecting logits strictly over a finite candidate set, output schemas are mathematically guaranteed. Formatting hallucinations and malformed JSON are impossible.
* **Direct C-API Logits Extraction**: Bypasses slow Python wrappers and high-level chat APIs by directly interfacing with prebuilt `llama.dll` through Python's `ctypes`. Reads raw float32 logits via `llama_get_logits_ith`.
* **Multi-Token Candidate Scoring**: Evaluates multi-token phrases (e.g. `"technical support"`) by accumulating conditional log-probabilities with length normalization ($\alpha$-penalty).
* **BoolQ Probability Calibration (RLCD-Style)**: Raw LLM logits are notoriously overconfident. OpenSourceJev includes empirical temperature scaling calibrated on **BoolQ** ($T \approx 9.4705$), cutting Expected Calibration Error (ECE) from **0.20 down to 0.09**!
* **The 4 Native Decision Primitives**:
  * **`Noul`** — Probabilistic boolean (`true` / `false`) answering *"Is this condition met?"* with calibrated probability.
  * **`Choice`** — Categorical classification selecting from a fixed set of options, with confidence scoring.
  * **`Score`** — Continuous mathematical expectation across ordered descriptive rubric levels.
  * **`Text`** — Optional short generated action or string continuation when needed.
* **Conditional Workflow Engine**: Chain steps together with conditional execution rules (e.g., `when: "urgent == true"`).
* **Interactive Web Playground**: Built-in browser UI with live execution traces, timing benchmarks, and candidate probability distributions.
* **Real-Time ViZDoom Agent**: Complete replication of the viral Jev DOOM controller, querying the engine in real time to aim, strafe, and shoot.

---

## 🛠️ System Requirements

* **Operating System**: Windows 10 / 11 (64-bit) or Linux x86_64
* **Python**: Python 3.11 or newer
* **GPU**: NVIDIA GPU with CUDA 12.x (e.g. RTX 3050 Laptop GPU or desktop GPU with $\ge$ 4 GB VRAM). CPU inference is also supported.
* **RAM**: 8 GB+ RAM

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sabeel111/OpenSourceJev.git
cd OpenSourceJev
```

### 2. Set Up Python Environment
Create and activate a virtual environment:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

### 3. Set Up the Native `llama.cpp` Runtime
The native engine uses prebuilt `llama.cpp` CUDA DLLs. On Windows with CUDA 12.4, run the automated setup script:

```powershell
.\install_native_cuda.ps1
```

This extracts the necessary runtime libraries into `runtime/llama.cpp/bin/`.

### 4. Download a Compatible GGUF Model
Download a GGUF model (e.g., **Qwen3-1.7B-Q8_0.gguf** or any Qwen2.5 / Qwen3 / Llama GGUF) and place it into the `models/` directory:

```text
models/Qwen3-1.7B-Q8_0.gguf
```

> *Tip: You can also specify an arbitrary external GGUF path in the UI or via the `JEV_LLAMA_MODEL` environment variable.*

---

## ⚡ Quickstart

### Launch the Web Playground & API

Start the engine using the launcher:

```powershell
.\run_mvp.ps1
```
*(Or double-click `run_mvp.bat`)*

Open your browser at **`http://127.0.0.1:8000`** to access the visual playground, inspect traces, and run test workflows.

Interactive Swagger API docs are available at **`http://127.0.0.1:8000/docs`**.

---

## 🎮 Real-Time ViZDoom Controller Demo

The repository includes a real-time game controller demonstrating OpenSourceJev's low-latency decision loop inside **ViZDoom**:

```
[ Game Memory (Health, Ammo, Enemy Angle) ]
                     │
                     ▼
             [ Tactical State ]
                     │
                     ▼
          [ OpenSourceJev Choice ]
      options: [shoot, strafe_left, strafe_right]
                     │
                     ▼
         [ Execute ViZDoom Action ]
```

### Run the DOOM Agent:
```powershell
cd doom_agent
.\run_doom.ps1
```

* Options:
  * `-NoWindow`: Runs headless in the terminal.
  * `-Episodes 5`: Specifies number of rounds.
  * `-Scenario defend_the_center`: Switch between survival and linear targeting scenarios.

---

## 💻 API & Python Usage

You can call OpenSourceJev via standard HTTP requests:

### Python Example:
```python
import requests

payload = {
    "context": (
        "Customer message: 'I was charged twice on invoice #48291. "
        "Please reverse the duplicate charge immediately!'"
    ),
    "mode": "native",
    "workflow": [
        {
            "id": "is_urgent",
            "kind": "noul",
            "prompt": "Does this request convey urgency?"
        },
        {
            "id": "category",
            "kind": "choice",
            "prompt": "Select the responsible department:",
            "options": ["billing", "technical_support", "account_access"]
        },
        {
            "id": "frustration_level",
            "kind": "score",
            "prompt": "Rate customer frustration on the rubric:",
            "criteria": [
                {"label": "Calm", "description": "Factual and neutral."},
                {"label": "Frustrated", "description": "Annoyed but polite."},
                {"label": "Angry", "description": "Demanding immediate action with exclamation."}
            ]
        },
        {
            "id": "escalation_action",
            "kind": "text",
            "prompt": "Write a 1-sentence action summary.",
            "when": "is_urgent == true"
        }
    ]
}

response = requests.post("http://127.0.0.1:8000/api/run", json=payload)
data = response.json()

print("Outputs:", data["outputs"])
# Outputs:
# {
#   "is_urgent": True,
#   "category": "billing",
#   "frustration_level": 1.84,
#   "escalation_action": "Route immediately to senior billing specialist for refund."
# }

# Access calibrated probabilities from the trace
for step in data["trace"]:
    if step["kind"] == "noul":
        print(f"Noul probability: {step['noul']:.4f}")
    if step["kind"] == "choice":
        print(f"Choice confidence: {step['confidence']:.4f}")
```

---

## 📊 Calibration Details

Raw transformer models output uncalibrated logits that cluster heavily at 0.0 or 1.0. OpenSourceJev implements **Temperature Scaling Calibration** based on experimental fits on the **BoolQ** benchmark:

| Metric | Raw Qwen3-1.7B | Calibrated ($T = 9.4705$) |
| :--- | :---: | :---: |
| **Accuracy** | **0.79** | **0.79** |
| **Negative Log-Likelihood (NLL)** | 2.3126 | **0.4781** |
| **Brier Score** | 0.1981 | **0.1522** |
| **Expected Calibration Error (ECE)** | 0.2021 | **0.0906** |

The fitted calibration metadata is stored in `jev_calibration.json` and applied automatically at runtime. You can override it via environment variables:
```powershell
$env:JEV_LLAMA_NOUL_TEMPERATURE = "9.470457"
```

---

## 🧪 Testing

Run the automated test suite with `pytest`:

```powershell
pytest
```

---

## 🗺️ Roadmap

- [x] Direct `llama.cpp` C API logits extraction via `ctypes`
- [x] Restricted candidate-only softmax for `Noul`, `Choice`, and `Score`
- [x] Multi-token candidate log-prob accumulation with length normalization
- [x] BoolQ temperature scaling calibration
- [x] FastAPI web playground & execution trace visualization
- [x] Real-time ViZDoom controller demo
- [ ] **Parallel KV-cache sequence copying (`llama_kv_cache_seq_cp`)** to evaluate $N$ independent questions simultaneously in $O(1)$ passes
- [ ] Multi-class calibration (Vector scaling / Matrix scaling) for `Choice`
- [ ] RLCD LoRA fine-tuning recipes for open-source models

---

## 📜 Acknowledgments & References

* **[TypeSafe AI](https://typesafe.ai)** for pioneering the System One decision model paradigm and introducing RLCD.
* **[Qwen Team (Alibaba Cloud)](https://github.com/QwenLM)** for the Qwen family of open-weight models.
* **[llama.cpp](https://github.com/ggerganov/llama.cpp)** by Georgi Gerganov for native inference infrastructure.
* **[ViZDoom](https://github.com/Farama-Foundation/ViZDoom)** for the Doom reinforcement learning platform.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
