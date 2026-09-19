import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .models import RunRequest, RunResponse, Step, StepTrace


class EngineError(RuntimeError):
    """A user-actionable workflow execution error."""


def _number(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _choice_options(step: Step) -> List[str]:
    if step.options:
        return list(step.options)
    if isinstance(step.criteria, dict):
        return list(step.criteria.keys())
    return []


def _strip_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = min([index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0], default=-1)
    if start > 0:
        cleaned = cleaned[start:]
    return cleaned


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(_strip_json(text))
    except json.JSONDecodeError as exc:
        raise EngineError(f"The model returned invalid JSON: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise EngineError("The model returned JSON, but it was not an object.")
    return parsed


def _parse_condition(condition: str, outputs: Dict[str, Any]) -> bool:
    """Evaluate the intentionally small, safe condition language used by workflows."""
    match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_-]*)\s*(==|!=|>=|<=|>|<)\s*(.*?)\s*", condition)
    if not match:
        raise EngineError(f"Unsupported condition '{condition}'. Use expressions like urgent == true.")

    key, operator, raw_expected = match.groups()
    if key not in outputs:
        return False

    expected: Any = raw_expected.strip().strip("'\"")
    lowered = expected.lower()
    if lowered in {"true", "false"}:
        expected = lowered == "true"
    else:
        try:
            expected = float(expected) if "." in expected else int(expected)
        except ValueError:
            pass

    actual = outputs[key]
    if operator == "==":
        return actual == expected or str(actual).lower() == str(expected).lower()
    if operator == "!=":
        return not (actual == expected or str(actual).lower() == str(expected).lower())
    try:
        if operator == ">=":
            return actual >= expected
        if operator == "<=":
            return actual <= expected
        if operator == ">":
            return actual > expected
        return actual < expected
    except TypeError as exc:
        raise EngineError(f"Cannot compare '{key}' with '{raw_expected}'.") from exc


def _mock_step(step: Step, context: str, outputs: Dict[str, Any]) -> Tuple[Any, float, List[Dict[str, Any]]]:
    # Keep the mock grounded in the supplied context. Including the question itself
    # would make a prompt such as "Is this urgent?" look like evidence of urgency.
    text = context.lower()

    if step.kind == "noul":
        negative = ("not " in text or "no " in text or "can't" in text or "cannot" in text
                    or "spam" in text)
        urgent = any(word in text for word in ("urgent", "critical", "outage", "blocked"))
        positive = any(word in text for word in ("qualified", "eligible", "business", "team plan", "demo", "approved"))
        value = urgent or (positive and not negative)
        return value, 0.88 if urgent else 0.76, [
            {"value": True, "score": 0.88 if value else 0.12},
            {"value": False, "score": 0.12 if value else 0.88},
        ]

    if step.kind == "choice":
        options = _choice_options(step)
        if not options:
            raise EngineError(f"Choice step '{step.id}' needs at least one option.")
        scored = []
        for option in options:
            normalized = option.lower()
            score = 0.2
            if normalized in text:
                score += 0.7
            keyword_groups = {
                "billing": ("payment", "invoice", "charge", "refund", "billing"),
                "technical": ("error", "crash", "bug", "outage", "failing", "broken"),
                "account": ("login", "password", "profile", "account"),
                "sales": ("buy", "plan", "pricing", "demo"),
            }
            for keyword in keyword_groups.get(normalized, ()):
                if keyword in text:
                    score += 0.2
            scored.append({"value": option, "score": round(score, 3)})
        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0]
        total = sum(item["score"] for item in scored) or 1
        return best["value"], round(best["score"] / total, 3), scored

    if step.kind == "score":
        if step.max <= step.min:
            raise EngineError(f"Score step '{step.id}' must have max greater than min.")
        score = 5.0
        if any(word in text for word in ("urgent", "critical", "outage", "blocked")):
            score += 3
        if any(word in text for word in ("error", "failing", "broken")):
            score += 1
        if any(word in text for word in ("low", "minor", "question")):
            score -= 2
        scaled = step.min + (step.max - step.min) * _clamp(score / 10, 0, 1)
        return round(scaled, 2), 0.72, []

    category = outputs.get("category", "the appropriate team")
    return f"Review the request and route it to {category} with the relevant context attached.", 0.68, []


class JevEngine:
    def __init__(self, ollama_url: Optional[str] = None):
        self.ollama_url = (ollama_url or os.getenv("JEV_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")

    async def run(self, request: RunRequest) -> RunResponse:
        if request.mode == "native":
            from .native_engine import NativeJevEngine

            return await NativeJevEngine().run(request)

        ids = [step.id for step in request.workflow]
        if len(ids) != len(set(ids)):
            raise EngineError("Every workflow step needs a unique id.")

        started = time.perf_counter()
        outputs: Dict[str, Any] = {}
        trace: List[StepTrace] = []

        for step in request.workflow:
            step_started = time.perf_counter()
            if step.when and not _parse_condition(step.when, outputs):
                trace.append(StepTrace(
                    id=step.id,
                    kind=step.kind,
                    status="skipped",
                    detail=f"Condition not met: {step.when}",
                    elapsed_ms=round((time.perf_counter() - step_started) * 1000),
                ))
                continue

            try:
                if request.mode == "mock":
                    value, confidence, candidates = _mock_step(step, request.context, outputs)
                else:
                    value, confidence, candidates = await self._ollama_step(
                        step, request.context, outputs, request.model, request.temperature
                    )
                outputs[step.id] = value
                trace.append(StepTrace(
                    id=step.id,
                    kind=step.kind,
                    status="completed",
                    value=value,
                    confidence=confidence,
                    candidates=candidates,
                    elapsed_ms=round((time.perf_counter() - step_started) * 1000),
                ))
            except EngineError as exc:
                trace.append(StepTrace(
                    id=step.id,
                    kind=step.kind,
                    status="error",
                    elapsed_ms=round((time.perf_counter() - step_started) * 1000),
                    detail=str(exc),
                ))
                return RunResponse(
                    status="error",
                    outputs=outputs,
                    trace=trace,
                    meta=self._meta(request, started),
                    error=str(exc),
                )

        return RunResponse(
            status="completed",
            outputs=outputs,
            trace=trace,
            meta=self._meta(request, started),
        )

    @staticmethod
    def _meta(request: RunRequest, started: float) -> Dict[str, Any]:
        return {
            "mode": request.mode,
            "model": request.model or os.getenv("JEV_OLLAMA_MODEL", "qwen3:1.7b"),
            "step_count": len(request.workflow),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _ollama_step(
        self,
        step: Step,
        context: str,
        outputs: Dict[str, Any],
        model: Optional[str],
        temperature: float,
    ) -> Tuple[Any, float, List[Dict[str, Any]]]:
        expected = {
            "noul": '{"value": true, "confidence": 0.0}',
            "choice": '{"value": "one option", "confidence": 0.0}',
            "score": '{"value": 0, "confidence": 0.0}',
            "text": '{"value": "one concise sentence", "confidence": 0.0}',
        }[step.kind]
        if step.kind == "choice":
            options = _choice_options(step)
            descriptions = step.criteria if isinstance(step.criteria, dict) else {}
            constraints = f"Criteria: {descriptions or options}"
        elif step.kind == "score" and isinstance(step.criteria, list):
            constraints = f"Ordered levels: {step.criteria}"
        else:
            constraints = f"Range: {step.min} to {step.max}" if step.kind == "score" else ""
        system = (
            "You are Jev, a local structured decision engine. Return only valid JSON with keys "
            "value and confidence. Confidence must be a number from 0 to 1. "
            f"For this step, the exact JSON shape is {expected}. {constraints}"
        )
        user = json.dumps({
            "request": context,
            "previous_outputs": outputs,
            "decision": step.prompt,
        }, ensure_ascii=False)

        payload = {
            "model": model or os.getenv("JEV_OLLAMA_MODEL", "qwen3:1.7b"),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise EngineError(
                f"Could not reach Ollama at {self.ollama_url}. Start Ollama or switch the mode to Demo / offline."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:240]
            raise EngineError(f"Ollama returned HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"Ollama request failed: {exc}") from exc

        content = data.get("message", {}).get("content", "")
        parsed = _parse_json(content)
        value = parsed.get("value")
        confidence = _clamp(_number(parsed.get("confidence"), 0.5), 0, 1)

        if step.kind == "noul":
            if isinstance(value, str):
                value = value.strip().lower() == "true"
            if not isinstance(value, bool):
                raise EngineError(f"Step '{step.id}' expected a boolean value.")
        elif step.kind == "choice":
            options = _choice_options(step)
            if value not in options:
                match = next((option for option in options if str(option).lower() == str(value).lower()), None)
                if match is None:
                    raise EngineError(f"Step '{step.id}' returned '{value}', which is not one of its options.")
                value = match
        elif step.kind == "score":
            if isinstance(step.criteria, list):
                if len(step.criteria) < 2:
                    raise EngineError(f"Score step '{step.id}' needs at least two ordered criteria levels.")
                value = round(_clamp(_number(value), 0, len(step.criteria) - 1), 2)
            else:
                if step.max <= step.min:
                    raise EngineError(f"Score step '{step.id}' must have max greater than min.")
                value = round(_clamp(_number(value), step.min, step.max), 2)
        elif not isinstance(value, str):
            value = str(value)

        candidates = [{"value": value, "score": round(confidence, 3)}]
        return value, round(confidence, 3), candidates
