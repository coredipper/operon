"""Tests for the improved judge prompt: rubric, token budget, and context limits.

Verifies that _judge_quality():
1. Sends a rubric-based prompt with score anchors
2. Uses adequate token budgets (≥150 for non-reasoning, ≥500 for reasoning)
3. Passes sufficient task/output context (≥800 chars task, ≥2000 chars output)
4. Still parses JSON and regex responses correctly
"""

from __future__ import annotations

from datetime import datetime

from operon_ai.providers.base import LLMResponse, ProviderConfig

import sys
sys.path.insert(0, ".")
from eval.convergence.live_evaluator import _judge_quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test",
        tokens_used=10,
        latency_ms=50.0,
        timestamp=datetime.now(),
    )


class _CapturingProvider:
    """Records the prompt and config passed to complete()."""

    def __init__(self, response: str = '{"score": 0.75}', model: str = "gpt-4o"):
        self._response = response
        self.model = model
        self.captured_prompt: str | None = None
        self.captured_config: ProviderConfig | None = None

    def is_available(self) -> bool:
        return True

    def complete(self, prompt, config=None):
        self.captured_prompt = prompt
        self.captured_config = config
        return _make_response(self._response)


# ---------------------------------------------------------------------------
# Tests: Rubric content
# ---------------------------------------------------------------------------


def test_prompt_contains_rubric_criteria():
    """Judge prompt must include correctness, completeness, and clarity criteria."""
    prov = _CapturingProvider()
    _judge_quality("Summarize this text", "A good summary", prov)

    prompt = prov.captured_prompt
    assert prompt is not None
    # All three rubric dimensions must appear
    assert "correctness" in prompt.lower(), "Missing 'correctness' in rubric"
    assert "completeness" in prompt.lower(), "Missing 'completeness' in rubric"
    assert "clarity" in prompt.lower(), "Missing 'clarity' in rubric"


def test_prompt_contains_score_anchors():
    """Judge prompt must include score band descriptions to anchor the scale."""
    prov = _CapturingProvider()
    _judge_quality("Summarize this text", "A good summary", prov)

    prompt = prov.captured_prompt
    assert prompt is not None
    # Must have at least low and high anchor descriptions
    assert "0.0" in prompt or "0.1" in prompt or "0.2" in prompt, \
        "Missing low-end score anchor"
    assert "0.8" in prompt or "0.9" in prompt or "1.0" in prompt, \
        "Missing high-end score anchor"


def test_prompt_contains_weight_guidance():
    """Judge prompt should indicate relative importance of criteria."""
    prov = _CapturingProvider()
    _judge_quality("Summarize this text", "A good summary", prov)

    prompt = prov.captured_prompt
    assert prompt is not None
    # Weight guidance — at least mention that correctness matters most
    assert "50%" in prompt or "most important" in prompt.lower() or \
        "weight" in prompt.lower(), \
        "Missing weight/importance guidance for rubric criteria"


# ---------------------------------------------------------------------------
# Tests: Token budget
# ---------------------------------------------------------------------------


def test_non_reasoning_model_gets_adequate_tokens():
    """Non-reasoning models should get ≥150 max_tokens (not 50)."""
    prov = _CapturingProvider(model="gpt-4o")
    _judge_quality("task", "output", prov)

    config = prov.captured_config
    assert config is not None
    assert config.max_tokens >= 150, \
        f"Non-reasoning model got only {config.max_tokens} tokens, need ≥150"


def test_reasoning_model_still_gets_500_tokens():
    """Reasoning models (deepseek-r1, qwen, nemotron) should get ≥500 max_tokens."""
    for model_name in ["deepseek-r1:8b", "qwen3:latest", "nvidia/nemotron-3-nano-4b",
                        "nemotron-3-nano-4b", "nvidia/llama-3.1-nemotron-70b"]:
        prov = _CapturingProvider(model=model_name)
        _judge_quality("task", "output", prov)

        config = prov.captured_config
        assert config is not None
        assert config.max_tokens >= 500, \
            f"Reasoning model {model_name} got only {config.max_tokens} tokens"


def test_gemini_gets_500_tokens_and_no_json_mode():
    """Gemini judges need ≥500 max_tokens and no response_format (JSON mode)."""
    from unittest.mock import patch
    from operon_ai.providers.gemini_provider import GeminiProvider

    # Stub GeminiProvider to capture config without real API call
    captured = {}

    def fake_complete(self, prompt, config=None):
        captured["config"] = config
        return _make_response('Here is my evaluation: {"score": 0.85}')

    with patch.object(GeminiProvider, "complete", fake_complete):
        g = GeminiProvider(model="gemini-2.5-flash")
        score, reason, _ = _judge_quality("Summarize this", "Good summary", g)

    assert captured["config"].max_tokens >= 500, \
        f"Gemini got {captured['config'].max_tokens} tokens, need ≥500"
    assert captured["config"].response_format is None, \
        "Gemini should not use JSON mode (response_format must be None)"
    assert score == 0.85, f"Expected 0.85 from embedded JSON, got {score}"


# ---------------------------------------------------------------------------
# Tests: Context limits
# ---------------------------------------------------------------------------


def test_task_truncation_at_least_800_chars():
    """Task context should allow at least 800 chars (up from 400)."""
    long_task = "A" * 1000
    prov = _CapturingProvider()
    _judge_quality(long_task, "output", prov)

    prompt = prov.captured_prompt
    # Count how many A's made it into the prompt
    a_count = prompt.count("A")
    assert a_count >= 800, \
        f"Task truncated to {a_count} chars, need ≥800"


def test_output_truncation_at_least_2000_chars():
    """Output context should allow at least 2000 chars (up from 1000)."""
    long_output = "B" * 3000
    prov = _CapturingProvider()
    _judge_quality("task", long_output, prov)

    prompt = prov.captured_prompt
    b_count = prompt.count("B")
    assert b_count >= 2000, \
        f"Output truncated to {b_count} chars, need ≥2000"


# ---------------------------------------------------------------------------
# Tests: Parsing — JSON primary, embedded JSON fallback, exact 0/1
# ---------------------------------------------------------------------------


def test_json_parsing():
    """Standard JSON response parses correctly."""
    prov = _CapturingProvider(response='{"score": 0.82}')
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 0.82


def test_embedded_json_in_prose():
    """JSON embedded in reasoning text is extracted."""
    prov = _CapturingProvider(response='I evaluated carefully. {"score": 0.75}')
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 0.75


def test_bare_decimal_json():
    """Bare '0.78' is valid JSON — parsed directly."""
    prov = _CapturingProvider(response="0.78")
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 0.78


def test_bare_one_point_zero():
    """Bare '1.0' is valid JSON — parsed directly."""
    prov = _CapturingProvider(response="1.0")
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 1.0


def test_embedded_json_with_whitespace():
    """JSON with spaces around colon is extracted."""
    prov = _CapturingProvider(response='Based on the rubric: { "score" : 0.60 }')
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 0.60


def test_embedded_json_subscores_ignored():
    """Only the {"score": N} JSON object matters, not surrounding numbers."""
    prov = _CapturingProvider(
        response='correctness=0.9, completeness=0.7. {"score": 0.78}'
    )
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 0.78


def test_exact_one():
    """Exact '1' as entire response parses."""
    prov = _CapturingProvider(response="1")
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 1.0


def test_exact_zero():
    """Exact '0' as entire response parses."""
    prov = _CapturingProvider(response="0")
    score, _, _ = _judge_quality("task", "output", prov)
    assert score == 0.0


def test_non_json_returns_unavailable():
    """Plain text without JSON returns 0.5 (triggers retry/fallback)."""
    for response in [
        "I think this is pretty good overall",
        "score: 0.85",  # no JSON braces
        "The score is 0.92 out of 1.0",
        "1 major issue remains",
        "overall: 0.7 out of 10",
        "10/10 great job",
        "correctness: 0.80\ncompleteness: 0.70\nscore: 0.75",
    ]:
        prov = _CapturingProvider(response=response)
        score, _, _ = _judge_quality("task", "output", prov)
        assert score == 0.5, f"Expected 0.5 for {response!r}, got {score}"
