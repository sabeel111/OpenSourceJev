import asyncio

from app.engine import JevEngine
from app.models import RunRequest


def test_mock_workflow_runs_and_skips_conditional_step():
    request = RunRequest(
        context="The checkout is failing for every customer and payments are blocked.",
        workflow=[
            {"id": "urgent", "kind": "noul", "prompt": "Is this urgent?"},
            {"id": "category", "kind": "choice", "prompt": "Classify it.", "options": ["billing", "technical"]},
            {"id": "severity", "kind": "score", "prompt": "Score from 0 to 10.", "min": 0, "max": 10},
            {"id": "action", "kind": "text", "prompt": "What next?", "when": "urgent == true"},
        ],
        mode="mock",
    )
    response = asyncio.run(JevEngine().run(request))
    assert response.status == "completed"
    assert response.outputs["urgent"] is True
    assert response.outputs["category"] == "billing"
    assert response.outputs["severity"] >= 5
    assert response.trace[-1].status == "completed"


def test_condition_can_skip_step():
    request = RunRequest(
        context="A small question.",
        workflow=[
            {"id": "urgent", "kind": "noul", "prompt": "Is this urgent?"},
            {"id": "action", "kind": "text", "prompt": "What next?", "when": "urgent == true"},
        ],
        mode="mock",
    )
    response = asyncio.run(JevEngine().run(request))
    assert response.status == "completed"
    assert response.trace[-1].status == "skipped"
    assert "action" not in response.outputs

