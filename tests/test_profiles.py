import os
from fastapi.testclient import TestClient

from app.main import app
from app.profiles import PROFILES, resolve_profile


def test_resolve_profile_default():
    profile = resolve_profile()
    assert profile.name in ("fast", "accuracy")


def test_resolve_profile_by_name():
    p_fast = resolve_profile(requested_profile="fast")
    assert p_fast.name == "fast"
    assert "1.7B" in p_fast.model_name
    assert abs(p_fast.default_temperature - 9.4705) < 0.01

    p_acc = resolve_profile(requested_profile="accuracy")
    assert p_acc.name == "accuracy"
    assert "4B" in p_acc.model_name
    assert abs(p_acc.default_temperature - 1.2364) < 0.01


def test_resolve_profile_by_model_alias():
    p_acc = resolve_profile(requested_model="qwen35-4b")
    assert p_acc.name == "accuracy"

    p_fast = resolve_profile(requested_model="qwen3-1.7b")
    assert p_fast.name == "fast"


def test_resolve_profile_by_env(monkeypatch):
    monkeypatch.setenv("JEV_PROFILE", "accuracy")
    p_acc = resolve_profile()
    assert p_acc.name == "accuracy"

    monkeypatch.setenv("JEV_PROFILE", "fast")
    p_fast = resolve_profile()
    assert p_fast.name == "fast"


def test_profile_provenance_structure():
    prof = PROFILES["accuracy"]
    prov = prof.to_provenance_dict()
    assert prov["profile"] == "accuracy"
    assert "Qwen3.5-4B" in prov["model_name"]
    assert "calibration_file" in prov
    assert "noul_temperature" in prov
    assert prov["prompt_style"] == "qwen3"


def test_api_v1_models_endpoint():
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data.get("object") == "list"
    models = data.get("data", [])
    model_ids = [m["id"] for m in models]
    assert any("1.7B" in m for m in model_ids)
    assert any("4B" in m for m in model_ids)


def test_api_systemone_returns_provenance():
    client = TestClient(app)
    payload = {
        "state": "The server CPU usage is 98%. Should we spin up a new instance?",
        "profile": "accuracy",
        "mode": "mock",
        "questions": {
            "scale_up": {
                "type": "noul",
                "instructions": "Should we scale up resources?"
            }
        }
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "answers" in res_data
    assert "scale_up" in res_data["answers"]
    assert "provenance" in res_data
    assert res_data["profile"] == "accuracy"
    assert "model_sha256" in res_data["provenance"]
