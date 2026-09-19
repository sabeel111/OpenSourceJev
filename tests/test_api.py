from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["app"] == "jev"


def test_run_endpoint():
    response = client.post("/api/run", json={
        "context": "The user cannot log in.",
        "workflow": [
            {"id": "category", "kind": "choice", "prompt": "Classify this.", "options": ["account", "billing"]}
        ],
        "mode": "mock",
    })
    assert response.status_code == 200
    assert response.json()["outputs"]["category"] == "account"


def test_native_mode_requires_a_gguf_path():
    response = client.post("/api/run", json={
        "context": "A test request.",
        "workflow": [{"id": "urgent", "kind": "noul", "prompt": "Is it urgent?"}],
        "mode": "native",
    })
    assert response.status_code == 400
    assert "GGUF path" in response.json()["detail"]
