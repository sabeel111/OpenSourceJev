"""Model Profile Management for OpenSourceJev.

Supports dynamic switching between:
1. Fast Mode (Default): Qwen3-1.7B-Q8_0 (312ms p50, minimal memory, 1.7GB)
2. Accuracy Mode: Qwen3.5-4B-Q4_K_M (436ms p50, +23.6% accuracy, 2.55GB)
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

_SHA256_CACHE: Dict[str, str] = {}


@dataclass
class ModelProfile:
    name: str
    model_name: str
    model_path: str
    calibration_path: str
    default_temperature: float
    expected_sha256: Optional[str]
    prompt_style: str = "qwen3"

    def is_available(self) -> bool:
        return Path(self.model_path).is_file()

    def get_sha256(self) -> str:
        canonical = str(Path(self.model_path).expanduser().resolve())
        if canonical in _SHA256_CACHE:
            return _SHA256_CACHE[canonical]
        if not Path(canonical).is_file():
            return "unknown_file_not_found"

        hasher = hashlib.sha256()
        with open(canonical, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        _SHA256_CACHE[canonical] = digest
        return digest

    def to_provenance_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.name,
            "model_name": self.model_name,
            "model_path": str(Path(self.model_path).expanduser().resolve()),
            "model_sha256": self.get_sha256(),
            "calibration_file": str(Path(self.calibration_path).expanduser().resolve()),
            "noul_temperature": self.default_temperature,
            "prompt_style": self.prompt_style,
            "backend": "llama.cpp (CUDA)",
        }


def _create_profiles() -> Dict[str, ModelProfile]:
    fast_path = str(ROOT_DIR / "models" / "Qwen3-1.7B-Q8_0.gguf")
    fast_cal = str(ROOT_DIR / "jev_calibration.json")
    fast_profile = ModelProfile(
        name="fast",
        model_name="Qwen/Qwen3-1.7B-Q8_0",
        model_path=fast_path,
        calibration_path=fast_cal,
        default_temperature=9.470457368922842,
        expected_sha256="061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
        prompt_style="qwen3",
    )

    acc_path = str(ROOT_DIR / "models" / "qwen35-4b-q4km" / "Qwen3.5-4B-Q4_K_M.gguf")
    acc_cal = str(ROOT_DIR / "jev_calibration_qwen35_4b_q4km.json")
    acc_profile = ModelProfile(
        name="accuracy",
        model_name="Qwen/Qwen3.5-4B-Q4_K_M",
        model_path=acc_path,
        calibration_path=acc_cal,
        default_temperature=1.2364066052728862,
        expected_sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        prompt_style="qwen3",
    )

    return {
        "fast": fast_profile,
        "accuracy": acc_profile,
    }


PROFILES: Dict[str, ModelProfile] = _create_profiles()

_PROFILE_ALIASES: Dict[str, str] = {
    "fast": "fast",
    "qwen3": "fast",
    "qwen3-1.7b": "fast",
    "1.7b": "fast",
    "default": "fast",
    "accuracy": "accuracy",
    "high_accuracy": "accuracy",
    "quality": "accuracy",
    "qwen35": "accuracy",
    "qwen3.5": "accuracy",
    "qwen35-4b": "accuracy",
    "qwen3.5-4b": "accuracy",
    "4b": "accuracy",
}


def resolve_profile(
    requested_profile: Optional[str] = None,
    requested_model: Optional[str] = None,
    model_path_override: Optional[str] = None,
) -> ModelProfile:
    """Resolve the active model profile according to request parameters, environment, and availability."""
    target_key = None

    # 1. Check explicit profile argument
    if requested_profile:
        norm = requested_profile.strip().lower()
        if norm in _PROFILE_ALIASES:
            target_key = _PROFILE_ALIASES[norm]

    # 2. Check explicit model argument
    if not target_key and requested_model:
        norm = requested_model.strip().lower()
        if norm in _PROFILE_ALIASES:
            target_key = _PROFILE_ALIASES[norm]
        elif "qwen35" in norm or "qwen3.5" in norm or "4b" in norm:
            target_key = "accuracy"
        elif "qwen3" in norm or "1.7b" in norm:
            target_key = "fast"

    # 3. Check environment override
    if not target_key:
        env_profile = os.getenv("JEV_PROFILE", "").strip().lower()
        if env_profile in _PROFILE_ALIASES:
            target_key = _PROFILE_ALIASES[env_profile]

    # 4. Check explicit model path override
    explicit_path = model_path_override or os.getenv("JEV_LLAMA_MODEL")
    if explicit_path:
        norm_path = explicit_path.lower().replace("\\", "/")
        if "qwen35" in norm_path or "qwen3.5" in norm_path:
            target_key = "accuracy"
        elif "qwen3" in norm_path or "1.7b" in norm_path:
            target_key = "fast"

    # Default to fast profile
    if not target_key:
        target_key = "fast"

    profile = PROFILES[target_key]

    # Graceful fallback if target model weight is missing on disk
    if not profile.is_available():
        other_key = "accuracy" if target_key == "fast" else "fast"
        other_profile = PROFILES[other_key]
        if other_profile.is_available():
            return other_profile

    # If explicit path override was given and exists, adopt it
    if explicit_path and Path(explicit_path).is_file():
        profile.model_path = str(Path(explicit_path).expanduser().resolve())

    return profile
