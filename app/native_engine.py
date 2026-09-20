"""Native Jev scoring through llama.cpp.

This module deliberately uses the low-level evaluation path rather than a chat
completion API. The model is evaluated to obtain the full next-token logits;
candidate sequences are then scored token by token under the same prefix.
"""

import asyncio
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .engine import EngineError, _parse_condition
from .llama_ctypes import CtypesLlama
from .models import RunRequest, RunResponse, Step, StepTrace


_MODEL_LOCKS: Dict[str, threading.RLock] = {}
_MODEL_POOL: Dict[Tuple[str, int, int, int, int], CtypesLlama] = {}
_POOL_LOCK = threading.RLock()
_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "jev_calibration.json"
_DEFAULT_NOUL_TEMPERATURE = 9.470457368922842


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _noul_temperature() -> float:
    """Return the recovered Colab calibration temperature for Noul."""
    configured = os.getenv("JEV_LLAMA_NOUL_TEMPERATURE")
    if configured is not None:
        return max(_env_float("JEV_LLAMA_NOUL_TEMPERATURE", _DEFAULT_NOUL_TEMPERATURE), 1e-6)
    try:
        data = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
        value = float(data["noul"]["temperature"])
        return max(value, 1e-6)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _DEFAULT_NOUL_TEMPERATURE


def _length_penalty_alpha() -> float:
    """Return the length normalization exponent for multi-token candidates."""
    configured = os.getenv("JEV_LENGTH_PENALTY_ALPHA")
    if configured is not None:
        return max(_env_float("JEV_LENGTH_PENALTY_ALPHA", 0.0), 0.0)
    return 0.0



def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise EngineError("llama.cpp returned an empty logits vector.")
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _log_softmax(values: Sequence[float]) -> List[float]:
    normalizer = _logsumexp(values)
    return [value - normalizer for value in values]


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _choice_options(step: Step) -> List[str]:
    if step.options:
        return list(step.options)
    if isinstance(step.criteria, dict):
        return list(step.criteria.keys())
    return []


def _score_levels(step: Step) -> Optional[List[Tuple[str, str]]]:
    if isinstance(step.criteria, list):
        if len(step.criteria) < 2:
            raise EngineError(f"Score step '{step.id}' needs at least two ordered criteria levels.")
        levels = []
        for index, item in enumerate(step.criteria):
            if isinstance(item, str):
                levels.append((str(index), item))
                continue
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("id")
                description = item.get("description") or item.get("meaning") or label
                if not label:
                    raise EngineError(f"Score step '{step.id}' level {index} needs a label.")
                levels.append((str(label), str(description)))
                continue
            raise EngineError(f"Score step '{step.id}' level {index} must be a string or object.")
        return levels
    return None


def _candidate_specs(step: Step) -> List[Tuple[Any, str]]:
    """Return semantic values and their literal continuations to score."""
    if step.kind == "noul":
        return [(True, " true"), (False, " false")]
    if step.kind == "choice":
        options = _choice_options(step)
        if not options:
            raise EngineError(f"Choice step '{step.id}' needs at least one option.")
        return [(option, f" {option}") for option in options]
    if step.kind == "score":
        levels = _score_levels(step)
        if levels is not None:
            scoring_mode = os.getenv("JEV_SCORE_SCORING_MODE", "index").lower()
            if scoring_mode == "text":
                return [(float(index), f" {description}") for index, (_label, description) in enumerate(levels)]
            return [(float(index), f" {label}") for index, (label, _description) in enumerate(levels)]
        if step.max <= step.min:
            raise EngineError(f"Score step '{step.id}' must have max greater than min.")
        span = step.max - step.min
        if span <= 20 and step.min.is_integer() and step.max.is_integer():
            values = [float(value) for value in range(int(step.min), int(step.max) + 1)]
        else:
            # A bounded score with a large/continuous range is represented by
            # eleven evenly spaced score buckets; the expected value is returned.
            values = [step.min + span * index / 10 for index in range(11)]
        return [(round(value, 4), f" {_format_number(value)}") for value in values]
    return []


def _criteria_text(step: Step) -> str:
    """Render the finite answer space in the same shape as the Colab engine."""
    if step.kind == "noul":
        if isinstance(step.criteria, dict):
            true_description = step.criteria.get("true", "Condition holds")
            false_description = step.criteria.get("false", "Condition does not hold")
            return (
                "Allowed answers:\n"
                f"- true: {json.dumps(true_description, ensure_ascii=False) if not isinstance(true_description, str) else true_description}\n"
                f"- false: {json.dumps(false_description, ensure_ascii=False) if not isinstance(false_description, str) else false_description}"
            )
        return "Allowed answers:\n- true\n- false"
    if step.kind == "choice":
        lines = ["Allowed choices:"]
        descriptions = step.criteria if isinstance(step.criteria, dict) else {}
        for option in _choice_options(step):
            description = descriptions.get(option)
            if isinstance(description, dict):
                description = description.get("description") or description.get("meaning") or description
            lines.append(f"- {option}" if not description else f"- {option}: {description}")
        return "\n".join(lines)
    if step.kind == "score":
        lines = ["Ordered score levels:"]
        levels = _score_levels(step)
        if levels is not None:
            for index, (label, description) in enumerate(levels):
                if label == str(index) or label == description:
                    lines.append(f"{index}: {description}")
                else:
                    lines.append(f"- {label}: {description}")
        else:
            lines.extend(f"- {_format_number(value)}" for value, _ in _candidate_specs(step))
        return "\n".join(lines)
    return "No fixed candidate set."


def _clean_generated_text(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL | re.IGNORECASE)
    return value.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


class _NativeModelPool:
    @staticmethod
    def get(model_path: str) -> CtypesLlama:
        path = str(Path(model_path).expanduser().resolve())
        if not Path(path).is_file():
            raise EngineError(f"Native GGUF model was not found: {path}")

        n_ctx = _env_int("JEV_LLAMA_N_CTX", 4096)
        n_batch = _env_int("JEV_LLAMA_N_BATCH", 512)
        n_threads = _env_int("JEV_LLAMA_N_THREADS", max((os.cpu_count() or 4) - 1, 1))
        n_gpu_layers = _env_int("JEV_LLAMA_N_GPU_LAYERS", -1)
        key = (path, n_ctx, n_batch, n_threads, n_gpu_layers)
        with _POOL_LOCK:
            if key not in _MODEL_POOL:
                try:
                    _MODEL_POOL[key] = CtypesLlama(
                        model_path=path,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers,
                    )
                except Exception as exc:
                    raise EngineError(f"Could not load native GGUF model '{path}': {exc}") from exc
            _MODEL_LOCKS.setdefault(path, threading.RLock())
            return _MODEL_POOL[key]

    @staticmethod
    def lock(model_path: str) -> threading.RLock:
        with _POOL_LOCK:
            return _MODEL_LOCKS.setdefault(model_path, threading.RLock())


class NativeJevEngine:
    """Actual Jev candidate scorer backed by llama.cpp logits."""

    async def run(self, request: RunRequest) -> RunResponse:
        return await asyncio.to_thread(self._run_sync, request)

    def _run_sync(self, request: RunRequest) -> RunResponse:
        ids = [step.id for step in request.workflow]
        if len(ids) != len(set(ids)):
            raise EngineError("Every workflow step needs a unique id.")

        model_path = request.model_path or os.getenv("JEV_LLAMA_MODEL")
        if not model_path:
            raise EngineError(
                "Native mode needs a GGUF path. Set JEV_LLAMA_MODEL or provide model_path in the request."
            )

        model = _NativeModelPool.get(model_path)
        canonical_path = str(Path(model_path).expanduser().resolve())
        started = time.perf_counter()
        outputs: Dict[str, Any] = {}
        trace: List[StepTrace] = []

        with _NativeModelPool.lock(canonical_path):
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
                    value, confidence, candidates = self._evaluate_step(
                        model, request.context, outputs, step, request.temperature
                    )
                    outputs[step.id] = value
                    noul_probability = None
                    if step.kind == "noul":
                        noul_probability = next(
                            (float(item["score"]) for item in candidates if item["value"] is True),
                            None,
                        )
                    trace.append(StepTrace(
                        id=step.id,
                        kind=step.kind,
                        status="completed",
                        value=value,
                        noul=noul_probability,
                        confidence=None if step.kind == "noul" else confidence,
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
                        meta=self._meta(request, canonical_path, started),
                        error=str(exc),
                    )

        return RunResponse(
            status="completed",
            outputs=outputs,
            trace=trace,
            meta=self._meta(request, canonical_path, started),
        )

    @staticmethod
    def _meta(request: RunRequest, model_path: str, started: float) -> Dict[str, Any]:
        return {
            "mode": "native",
            "backend": "llama.cpp",
            "model_path": model_path,
            "prompt_style": os.getenv("JEV_LLAMA_PROMPT_STYLE", "qwen3").lower(),
            "noul_temperature": _noul_temperature(),
            "length_penalty_alpha": _length_penalty_alpha(),
            "step_count": len(request.workflow),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "timestamp": time.time(),
        }

    def _evaluate_step(
        self,
        model: Any,
        context: str,
        outputs: Dict[str, Any],
        step: Step,
        temperature: float,
    ) -> Tuple[Any, Optional[float], List[Dict[str, Any]]]:
        prefix = self._decision_prefix(context, outputs, step)
        if step.kind == "text":
            return self._generate_text(model, prefix, temperature), None, []

        specs = _candidate_specs(step)
        scored = self._score_candidates(model, prefix, specs)
        alpha = _length_penalty_alpha() if step.kind in {"choice", "score"} else 0.0
        raw_probabilities = self._candidate_probabilities(scored, alpha=alpha)
        calibration_temperature = _noul_temperature() if step.kind == "noul" else 1.0
        probabilities = self._candidate_probabilities(scored, calibration_temperature, alpha=alpha)
        candidates = []
        for index, ((value, _text, logprob, token_count), probability) in enumerate(zip(scored, probabilities)):
            candidate = {
                "value": value,
                "score": round(probability, 6),
                "logprob": round(logprob, 6),
                "token_count": token_count,
            }
            if step.kind in {"choice", "score"}:
                if step.kind == "score" and _score_levels(step) is not None:
                    levels = _score_levels(step)
                    candidate["label"] = levels[index][1] if levels else _text.strip()
                else:
                    candidate["label"] = _text.strip()
            if step.kind == "noul":
                candidate["raw_score"] = round(raw_probabilities[index], 6)
                candidate["calibrated_score"] = round(probability, 6)
                candidate["calibration_temperature"] = calibration_temperature
            candidates.append(candidate)

        if step.kind == "score":
            value = round(sum(float(item["value"]) * item["score"] for item in candidates), 4)
            confidence = max(probabilities)
            return value, round(confidence, 6), candidates

        best_index = max(range(len(candidates)), key=lambda index: candidates[index]["score"])
        return candidates[best_index]["value"], candidates[best_index]["score"], candidates

    @staticmethod
    def _decision_prefix(context: str, outputs: Dict[str, Any], step: Step) -> str:
        previous = str(outputs) if outputs else "{}"
        state = f"Context:\n{context}"
        if outputs:
            state += f"\n\nPrevious outputs:\n{previous}"
        body = (
            f"STATE:\n{state}\n\n"
            f"QUESTION TYPE:\n{step.kind}\n\n"
            f"INSTRUCTIONS:\n{step.prompt}\n\n"
            f"CRITERIA:\n{_criteria_text(step)}"
        )
        if step.kind == "score" and _score_levels(step) is not None:
            max_idx = len(_score_levels(step)) - 1
            body += f"\n\nSelect the score level number (0 to {max_idx}) that best matches."
        elif step.kind == "noul":
            body += "\n\nAnswer with true or false."

        style = os.getenv("JEV_LLAMA_PROMPT_STYLE", "qwen3").lower()
        if style == "qwen3":
            return (
                "<|im_start|>user\n"
                f"{body}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                "ANSWER:"
            )
        instruction = (
            "You are a structured decision engine. Evaluate the decision and output only the answer value. "
            "Do not explain your reasoning. Do not output JSON. Do not output a thinking block."
        )
        return f"System:\n{instruction}\n\n{body}\n\nAssistant:\nANSWER:"

    @staticmethod
    def _prefix_tokens(model: Any, prefix: str) -> List[int]:
        try:
            tokens = model.tokenize(prefix, add_special=True, parse_special=True)
        except TypeError:
            # Keep the tiny fake model used by the unit tests compatible.
            tokens = model.tokenize(prefix.encode("utf-8"), add_bos=True, special=True)
        except Exception as exc:
            raise EngineError(f"Native tokenizer failed: {exc}") from exc
        if not tokens:
            raise EngineError("Native tokenizer produced an empty decision prefix.")
        return list(tokens)

    @staticmethod
    def _last_logits(model: Any) -> List[float]:
        try:
            logits = model.last_logits()
        except AttributeError:
            try:
                logits = model.eval_logits[-1]
            except Exception as exc:
                raise EngineError(f"llama.cpp did not return next-token logits: {exc}") from exc
        except Exception as exc:
            raise EngineError(f"llama.cpp did not return next-token logits: {exc}") from exc
        if not logits:
            raise EngineError("llama.cpp returned an empty next-token logits vector.")
        return [float(value) for value in logits]

    def _score_candidates(
        self, model: Any, prefix: str, specs: Iterable[Tuple[Any, str]]
    ) -> List[Tuple[Any, str, float, int]]:
        prefix_tokens = self._prefix_tokens(model, prefix)
        prefix_len = len(prefix_tokens)
        try:
            model.reset()
            model.eval(prefix_tokens)
        except Exception as exc:
            raise EngineError(f"llama.cpp failed to evaluate the decision prefix: {exc}") from exc

        # Read the logits immediately following the shared prefix.
        prefix_logits = self._last_logits(model)
        prefix_log_probs = _log_softmax(prefix_logits)

        can_trim = hasattr(model, "trim_kv") and getattr(model, "supports_trim", lambda: True)()
        alpha = _length_penalty_alpha()

        results: List[Tuple[Any, str, float, int]] = []

        for value, candidate_text in specs:
            try:
                candidate_tokens = list(model.tokenize(candidate_text, add_special=False, parse_special=False))
            except TypeError:
                candidate_tokens = list(model.tokenize(candidate_text.encode("utf-8"), add_bos=False, special=False))
            except Exception as exc:
                raise EngineError(f"Native tokenizer failed for candidate '{value}': {exc}") from exc
            if not candidate_tokens:
                raise EngineError(f"Candidate '{value}' produced no tokens.")

            n_tokens = len(candidate_tokens)
            first_token = candidate_tokens[0]
            if first_token < 0 or first_token >= len(prefix_logits):
                raise EngineError(f"Candidate '{value}' produced an invalid token id {first_token}.")

            # The first token's logprob comes directly from the prefix logits without extra evaluation
            logprob = prefix_log_probs[first_token]

            # If multi-token, evaluate the intermediate tokens
            if n_tokens > 1:
                evaluated_extra = False
                try:
                    for position in range(1, n_tokens):
                        model.eval([candidate_tokens[position - 1]])
                        evaluated_extra = True
                        step_logits = self._last_logits(model)
                        next_token = candidate_tokens[position]
                        if next_token < 0 or next_token >= len(step_logits):
                            raise EngineError(f"Candidate '{value}' produced an invalid token id {next_token}.")
                        logprob += _log_softmax(step_logits)[next_token]
                finally:
                    if evaluated_extra:
                        if can_trim:
                            try:
                                model.trim_kv(prefix_len)
                            except Exception:
                                model.reset()
                                model.eval(prefix_tokens)
                        else:
                            model.reset()
                            model.eval(prefix_tokens)

            results.append((value, candidate_text, logprob, n_tokens))

        return results

    @staticmethod
    def _candidate_probabilities(
        scored: Sequence[Tuple[Any, str, float, int]], temperature: float = 1.0, alpha: float = 0.0
    ) -> List[float]:
        if temperature <= 0:
            raise EngineError("Candidate probability temperature must be greater than zero.")
        if alpha > 0:
            logprobs = [
                (item[2] / (item[3] ** alpha) if item[3] > 1 else item[2]) / temperature
                for item in scored
            ]
        else:
            logprobs = [item[2] / temperature for item in scored]
        normalizer = _logsumexp(logprobs)
        return [math.exp(logprob - normalizer) for logprob in logprobs]

    @staticmethod
    def _generate_text(model: Any, prefix: str, temperature: float) -> str:
        try:
            text = model.generate(prefix, max(temperature, 0.2), max_tokens=128)
        except Exception as exc:
            raise EngineError(f"Native text generation failed: {exc}") from exc
        cleaned = _clean_generated_text(str(text))
        if not cleaned:
            raise EngineError("Native text generation returned an empty answer.")
        return cleaned
