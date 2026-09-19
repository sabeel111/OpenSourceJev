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
        "id": "support_triage",
        "name": "Support triage",
        "description": "Route a customer issue and decide whether it needs an urgent response.",
        "context": "The checkout is failing for every customer and payments are blocked.",
        "workflow": [
            {"id": "urgent", "kind": "noul", "prompt": "Is this request urgent?"},
            {"id": "category", "kind": "choice", "prompt": "Classify the request.", "options": ["billing", "technical", "account", "sales"]},
            {"id": "severity", "kind": "score", "prompt": "Score the severity from 0 to 10.", "min": 0, "max": 10},
            {"id": "next_action", "kind": "text", "prompt": "Write the next action in one concise sentence.", "when": "urgent == true"},
        ],
    },
    {
        "id": "lead_qualification",
        "name": "Lead qualification",
        "description": "Use a boolean gate, a category, and a bounded fit score.",
        "context": "A startup with 30 engineers wants a team plan and asked for a product demo next week.",
        "workflow": [
            {"id": "qualified", "kind": "noul", "prompt": "Is this a qualified business lead?"},
            {"id": "segment", "kind": "choice", "prompt": "Choose the best segment.", "options": ["startup", "mid-market", "enterprise", "consumer"]},
            {"id": "fit_score", "kind": "score", "prompt": "Score product fit from 0 to 100.", "min": 0, "max": 100},
            {"id": "follow_up", "kind": "text", "prompt": "Suggest a short follow-up action.", "when": "qualified == true"},
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
