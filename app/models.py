from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


StepKind = Literal["noul", "choice", "score", "text"]
RunMode = Literal["mock", "ollama", "native"]


class Step(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    kind: StepKind
    prompt: str = Field(min_length=1, max_length=2000)
    options: List[str] = Field(default_factory=list, max_length=256)
    # Jev-style criteria: option -> description for Choice/Noul, or an
    # ordered list of level descriptions for Score.
    criteria: Optional[Union[Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = None
    min: float = 0
    max: float = 10
    when: Optional[str] = Field(default=None, max_length=200)


class RunRequest(BaseModel):
    context: str = Field(min_length=1, max_length=20000)
    workflow: List[Step] = Field(min_length=1, max_length=32)
    mode: RunMode = "mock"
    model: Optional[str] = Field(default=None, max_length=120)
    model_path: Optional[str] = Field(default=None, max_length=1000)
    temperature: float = Field(default=0.0, ge=0, le=1)


class StepTrace(BaseModel):
    id: str
    kind: StepKind
    status: Literal["completed", "skipped", "error"]
    value: Any = None
    # Official Jev-style Noul probability that the answer is true/yes.
    # The workflow output remains boolean for backwards-compatible conditions.
    noul: Optional[float] = None
    confidence: Optional[float] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: int = 0
    detail: Optional[str] = None


class RunResponse(BaseModel):
    status: Literal["completed", "error"]
    outputs: Dict[str, Any] = Field(default_factory=dict)
    trace: List[StepTrace] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
