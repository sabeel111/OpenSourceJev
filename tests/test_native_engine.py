import math

from app.native_engine import (
    NativeJevEngine,
    _candidate_specs,
    _criteria_text,
    _log_softmax,
    _noul_temperature,
)
from app.models import Step


def test_native_log_softmax_is_normalized():
    values = _log_softmax([0.0, 1.0, 2.0])
    probabilities = [math.exp(value) for value in values]
    assert abs(sum(probabilities) - 1.0) < 1e-9
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_native_choice_candidates_preserve_multi_token_values():
    step = Step(id="route", kind="choice", prompt="Pick a route.", options=["technical support", "billing"])
    specs = _candidate_specs(step)
    assert specs == [("technical support", " technical support"), ("billing", " billing")]


def test_native_score_uses_integer_support_for_small_ranges():
    step = Step(id="severity", kind="score", prompt="Score it.", min=0, max=2)
    specs = _candidate_specs(step)
    assert [value for value, _ in specs] == [0.0, 1.0, 2.0]


def test_native_qwen3_prefix_matches_colab_non_thinking_shape(monkeypatch):
    monkeypatch.setenv("JEV_LLAMA_PROMPT_STYLE", "qwen3")
    step = Step(id="urgent", kind="noul", prompt="Does this require escalation?")
    prefix = NativeJevEngine._decision_prefix("A service is down.", {}, step)

    assert prefix.startswith("<|im_start|>user\nSTATE:")
    assert "QUESTION TYPE:\nnoul" in prefix
    assert "Allowed answers:\n- true\n- false" in prefix
    assert prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_recovered_noul_temperature_softens_confidence():
    scored = [(True, " true", 4.0, 1), (False, " false", 0.0, 1)]
    raw = NativeJevEngine._candidate_probabilities(scored)
    calibrated = NativeJevEngine._candidate_probabilities(scored, _noul_temperature())

    assert _noul_temperature() > 1.0
    assert raw[0] > calibrated[0] > 0.5
    assert abs(sum(calibrated) - 1.0) < 1e-9


def test_native_criteria_rendering_matches_typed_question_shape():
    noul = Step(id="gate", kind="noul", prompt="Is it valid?")
    choice = Step(id="route", kind="choice", prompt="Choose a route.", options=["billing", "technical"])
    score = Step(id="severity", kind="score", prompt="Score it.", min=0, max=2)

    assert _criteria_text(noul) == "Allowed answers:\n- true\n- false"
    assert "- billing" in _criteria_text(choice)
    assert _criteria_text(score).endswith("- 0\n- 1\n- 2")


def test_native_jev_style_criteria_supports_descriptive_levels():
    step = Step(
        id="severity",
        kind="score",
        prompt="How severe is the issue?",
        criteria=[
            "Isolated user error; no service impact",
            "Degraded feature; workaround exists",
            "Blocking issue; no workaround exists",
        ],
    )
    specs = _candidate_specs(step)

    assert [value for value, _ in specs] == [0.0, 1.0, 2.0]
    assert specs[0][1] == " Isolated user error; no service impact"
    assert "- Blocking issue; no workaround exists" in _criteria_text(step)


def test_native_score_separates_short_answer_labels_from_descriptions():
    step = Step(
        id="severity",
        kind="score",
        prompt="How severe is the issue?",
        criteria=[
            {"label": "Minor", "description": "Isolated user error; no service impact."},
            {"label": "Major", "description": "Several customers affected; meaningful disruption."},
        ],
    )
    specs = _candidate_specs(step)

    assert [text for _, text in specs] == [" Minor", " Major"]
    assert "- Minor: Isolated user error; no service impact." in _criteria_text(step)


class FakeLlama:
    """Tiny deterministic model-shaped object for testing branch scoring."""

    def __init__(self):
        self.tokens = []
        self.eval_logits = [[0.0, 0.0, 0.0]]

    def tokenize(self, text, add_special=True, parse_special=False, add_bos=True, special=False):
        if isinstance(text, str):
            text = text.encode()
        mapping = {b"prefix": [0], b" a": [1], b" ab": [1, 2], b" b": [2]}
        return mapping[text]

    def reset(self):
        self.tokens = []

    def eval(self, tokens):
        self.tokens.extend(tokens)
        if self.tokens == [0]:
            self.eval_logits = [[0.0, 4.0, 1.0]]
        elif self.tokens == [0, 1]:
            self.eval_logits = [[0.0, 0.0, 5.0]]

    def last_logits(self):
        return self.eval_logits[-1]

    def save_state(self):
        return list(self.tokens), [list(row) for row in self.eval_logits]

    def load_state(self, state):
        self.tokens, self.eval_logits = list(state[0]), [list(row) for row in state[1]]


def test_native_scores_multi_token_candidates_from_next_token_logits():
    model = FakeLlama()
    engine = NativeJevEngine()
    scored = engine._score_candidates(model, "prefix", [("a", " a"), ("ab", " ab")])
    assert scored[0][2] > scored[1][2]
    assert scored[1][3] == 2


def test_native_length_normalization_softens_multitoken_penalty():
    # Candidate 1: 1 token with logprob -0.50
    # Candidate 2: 3 tokens with raw logprob -0.60 (3 x -0.20 each)
    scored = [("one", " one", -0.50, 1), ("three", " three words here", -0.60, 3)]
    probs_raw = NativeJevEngine._candidate_probabilities(scored, alpha=0.0)
    assert probs_raw[0] > probs_raw[1]

    probs_normalized = NativeJevEngine._candidate_probabilities(scored, alpha=0.7)
    # With alpha=0.7, -0.60 / (3^0.7) = -0.60 / 2.158 = -0.278, which is higher than -0.50
    assert probs_normalized[1] > probs_normalized[0]


class TrimSpyLlama(FakeLlama):
    def __init__(self):
        super().__init__()
        self.trims = []

    def supports_trim(self):
        return True

    def trim_kv(self, pos):
        self.trims.append(pos)
        return True


def test_native_trim_kv_is_invoked_when_supported():
    model = TrimSpyLlama()
    engine = NativeJevEngine()
    scored = engine._score_candidates(model, "prefix", [("a", " a"), ("ab", " ab")])
    assert len(model.trims) == 1
    assert model.trims[0] == 1
    assert scored[1][3] == 2

