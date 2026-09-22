import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import EngineError, JevEngine
from .llama_ctypes import native_status
from .models import RunRequest, RunResponse


ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = os.getenv("JEV_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
app = FastAPI(title="Jev MVP", version="0.1.0", description="Local structured decision playground")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


EXAMPLES = [
    {
        "id": "security_gate",
        "name": "Safe Exec (Security Guardrail)",
        "description": "Evaluate shell commands or tool calls for destructive operations, secrets leakage, and required human sign-off.",
        "context": "rm -rf /var/log/app/* && drop database production_db;",
        "workflow": [
            {"id": "is_destructive", "kind": "noul", "prompt": "Does this command contain destructive, irreversible, or data-loss operations?"},
            {"id": "accesses_secrets", "kind": "noul", "prompt": "Does this command access or leak credentials, tokens, or private keys?"},
            {"id": "risk_tier", "kind": "score", "prompt": "Score the risk level from 1 (read-only safe) to 5 (critical/destructive).", "min": 1, "max": 5},
            {"id": "block_command", "kind": "noul", "prompt": "Should this command be blocked from execution?", "when": "is_destructive == true"},
        ],
    },
    {
        "id": "multi_agent_router",
        "name": "Multi-Agent Intent Router",
        "description": "Speculatively fan-out classification across specialized agents with urgency and sentiment scoring in one pass.",
        "context": "Our production PostgreSQL database is throwing connection pool timeout errors during peak checkout. Can you run EXPLAIN ANALYZE on query 492 and diagnose why it is slow?",
        "workflow": [
            {"id": "target_agent", "kind": "choice", "prompt": "Select the specialized sub-agent best suited to resolve this issue.", "options": ["database_specialist", "devops_infra", "billing_support", "code_refactor"]},
            {"id": "urgency_score", "kind": "score", "prompt": "Score the urgency from 1 to 10 based on production business impact.", "min": 1, "max": 10},
            {"id": "sentiment", "kind": "choice", "prompt": "Detect the user's emotional state.", "options": ["calm", "neutral", "frustrated", "urgent"]},
            {"id": "page_oncall", "kind": "noul", "prompt": "Should this ticket immediately page an on-call engineer?", "when": "urgency_score >= 8"},
        ],
    },
    {
        "id": "judge_jev",
        "name": "Judge Jev (RAG Relevance)",
        "description": "High-speed semantic validation of retrieved context chunks and hallucination risk before generative synthesis.",
        "context": "User Query: 'What is TypeSafe Jev?' | Retrieved Document: 'Jev is a non-autoregressive System One decision model by TypeSafe AI that outputs calibrated typed decisions in sub-50ms forward passes rather than generating freeform text.'",
        "workflow": [
            {"id": "is_relevant", "kind": "noul", "prompt": "Is the retrieved document chunk directly relevant and sufficient to answer the user query?"},
            {"id": "hallucination_risk", "kind": "noul", "prompt": "Does the document contain speculative or ungrounded claims?"},
            {"id": "relevance_score", "kind": "score", "prompt": "Rate the factual relevance quality of this chunk from 1 to 10.", "min": 1, "max": 10},
        ],
    },
    {
        "id": "support_triage",
        "name": "Enterprise Support Triage",
        "description": "Route a customer issue, assess SLA severity, and determine immediate operational protocol.",
        "context": "The checkout is failing for every customer on the enterprise plan and credit card authorizations are failing.",
        "workflow": [
            {"id": "urgent", "kind": "noul", "prompt": "Is this request urgent and SLA-critical?"},
            {"id": "category", "kind": "choice", "prompt": "Classify the domain of the request.", "options": ["billing", "technical", "account", "security"]},
            {"id": "severity", "kind": "score", "prompt": "Score the severity from 0 to 10.", "min": 0, "max": 10},
            {"id": "protocol", "kind": "choice", "prompt": "Select the immediate response protocol.", "options": ["page_incident_commander", "route_to_tier_2", "schedule_followup", "auto_refund"], "when": "urgent == true"},
        ],
    },
    {
        "id": "lead_qualification",
        "name": "B2B Lead Qualification",
        "description": "Qualify business leads with bounded fit scores, market segmenting, and sales routing.",
        "context": "A Series B enterprise with 65 engineers wants dedicated GPU inference on VPC and requested an architectural review next week.",
        "workflow": [
            {"id": "qualified", "kind": "noul", "prompt": "Is this a qualified enterprise business lead?"},
            {"id": "segment", "kind": "choice", "prompt": "Choose the best market segment.", "options": ["seed_startup", "growth_scaleup", "enterprise", "consumer"]},
            {"id": "fit_score", "kind": "score", "prompt": "Score product architecture fit from 0 to 100.", "min": 0, "max": 100},
            {"id": "routing", "kind": "choice", "prompt": "Select routing path.", "options": ["executive_intro", "inbound_ae", "self_serve_docs"], "when": "qualified == true"},
        ],
    },
]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    ollama = {"connected": False, "url": OLLAMA_URL, "models": []}
    try:
        async with httpx.AsyncClient(timeout=0.8) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            ollama["connected"] = True
            ollama["models"] = [item.get("name") for item in data.get("models", []) if item.get("name")]
    except httpx.HTTPError:
        pass
    native = native_status()
    native["gpu_layers"] = os.getenv("JEV_LLAMA_N_GPU_LAYERS", "-1")
    bundled_model = ROOT / "models" / "Qwen3-1.7B-Q8_0.gguf"
    native["default_model"] = os.getenv(
        "JEV_LLAMA_MODEL",
        str(bundled_model) if bundled_model.is_file() else "",
    )
    return {
        "status": "ok",
        "app": "jev",
        "version": app.version,
        "ollama": ollama,
        "native": native,
        "default_model": os.getenv("JEV_OLLAMA_MODEL", "qwen3:1.7b"),
    }


@app.get("/api/examples")
async def examples() -> Dict[str, Any]:
    return {"examples": EXAMPLES}


@app.post("/api/run", response_model=RunResponse)
async def run_workflow(request: RunRequest) -> RunResponse:
    try:
        return await JevEngine().run(request)
    except EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/systemone")
async def systemone(payload: Dict[str, Any]) -> Dict[str, Any]:
    """TypeSafe Jev wire protocol compatibility endpoint (POST /v1/systemone).

    Allows official TypeSafe SDKs, LiteLLM, JevBench harnesses, and community
    adapters to query OpenSourceJev as a drop-in replacement for hosted Jev.
    """
    raw_state = payload.get("state", "")
    if isinstance(raw_state, (dict, list)):
        state_str = json.dumps(raw_state, ensure_ascii=False)
    else:
        state_str = str(raw_state)

    questions_map = payload.get("questions") or payload.get("schema") or {}
    if not questions_map:
        raise HTTPException(status_code=400, detail="Request missing 'questions' or 'schema' map.")

    from .models import Step
    workflow = []
    for q_id, q_data in questions_map.items():
        q_type = q_data.get("type", "choice").lower()
        instructions = q_data.get("instructions") or q_data.get("prompt") or ""
        criteria = q_data.get("criteria", None)
        options = q_data.get("options", [])

        if q_type == "choice" and not options:
            if isinstance(criteria, dict):
                options = list(criteria.keys())
            elif isinstance(criteria, list):
                options = [str(item) for item in criteria]

        step = Step(
            id=q_id,
            kind=q_type,
            prompt=instructions,
            options=options,
            criteria=criteria,
            min=float(q_data.get("min", 0)),
            max=float(q_data.get("max", 10)),
        )
        workflow.append(step)

    req_model = payload.get("model")
    bundled_model = ROOT / "models" / "Qwen3-1.7B-Q8_0.gguf"
    qwen35_model = ROOT / "models" / "qwen35-4b-q4km" / "Qwen3.5-4B-Q4_K_M.gguf"
    if req_model and ("qwen35" in req_model.lower() or "qwen3.5" in req_model.lower()) and qwen35_model.is_file():
        default_model = str(qwen35_model)
    else:
        default_model = str(bundled_model) if bundled_model.is_file() else None
    model_path = os.getenv("JEV_LLAMA_MODEL", default_model)

    req = RunRequest(
        context=state_str,
        workflow=workflow,
        mode="native" if model_path else "mock",
        model_path=model_path,
    )

    try:
        resp = await JevEngine().run(req)
    except EngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    answers: Dict[str, Any] = {}
    for trace in resp.trace:
        q_id = trace.id
        kind = trace.kind
        if kind == "choice":
            probs = {c["value"]: c["score"] for c in trace.candidates}
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values()) if probs else 0.0
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
            certainty = round(1.0 - (entropy / max_entropy), 4) if max_entropy > 0 else 1.0
            answers[q_id] = {
                "type": "choice",
                "choice": trace.value,
                "confidence": trace.confidence,
                "certainty": certainty,
                "probabilities": probs,
            }
        elif kind == "noul":
            answers[q_id] = {
                "type": "noul",
                "noul": trace.noul if trace.noul is not None else 0.5,
            }
        elif kind == "score":
            prob_dict = {}
            for c in trace.candidates:
                val = c["value"]
                if isinstance(val, (int, float)) and float(val).is_integer():
                    key_str = str(int(val))
                else:
                    key_str = str(val)
                prob_dict[key_str] = c["score"]

            answers[q_id] = {
                "type": "score",
                "score": trace.value,
                "confidence": trace.confidence,
                "probabilities": prob_dict,
            }
        else:
            answers[q_id] = {
                "type": kind,
                "value": trace.value,
            }

    return {
        "model": payload.get("model", "opensourcejev-qwen3-1.7b"),
        "answers": answers,
        "usage": {
            "input_tokens": resp.meta.get("input_tokens", max(1, len(state_str) // 4)),
            "output_tokens": len(workflow),
        },
    }
