"""Regression tests for the judge fallback behavior.

Verifies that:
1. _judge_quality() does NOT call fallback_factory when the primary succeeds
2. _judge_quality() calls fallback_factory exactly once when primary exhausts retries
3. LiveEvaluator.evaluate_task() does NOT resolve self.ollama when a remote judge succeeds
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, PropertyMock

from operon_ai.providers.base import LLMResponse, ProviderConfig

import sys
sys.path.insert(0, ".")
from eval.convergence.live_evaluator import _judge_quality, LiveEvaluator
from operon_ai.providers.gemini_provider import GeminiProvider


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


class _SuccessProvider:
    """Provider that always returns a valid JSON score."""
    model = "test-success"
    def is_available(self) -> bool: return True
    def complete(self, prompt, config=None):
        return _make_response('{"score": 0.85}')


class _FailProvider:
    """Provider that always raises."""
    model = "test-fail"
    def is_available(self) -> bool: return True
    def complete(self, prompt, config=None):
        raise ConnectionError("simulated failure")


class _FallbackProvider:
    """Provider used as fallback — tracks whether it was called."""
    model = "test-fallback"
    call_count = 0
    def is_available(self) -> bool: return True
    def complete(self, prompt, config=None):
        self.call_count += 1
        return _make_response('{"score": 0.70}')


# Patch time.sleep to avoid real delays in failure-path tests
_SLEEP_PATCH = "eval.convergence.live_evaluator.time.sleep"


# ---------------------------------------------------------------------------
# Tests: _judge_quality fallback behavior
# ---------------------------------------------------------------------------


def test_no_fallback_when_primary_succeeds():
    """Fallback factory should NOT be called when the primary provider succeeds."""
    factory_called = False

    def factory():
        nonlocal factory_called
        factory_called = True
        return _FallbackProvider()

    score, _, _ = _judge_quality(
        "Summarize this text",
        "Good summary here",
        _SuccessProvider(),
        fallback_factory=factory,
    )

    assert score == 0.85
    assert not factory_called, "Fallback factory was called despite primary success"


@patch(_SLEEP_PATCH)
def test_fallback_called_after_primary_failure(_sleep):
    """Fallback factory should be called exactly once when primary exhausts retries."""
    factory_calls = 0
    fb = _FallbackProvider()

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return fb

    score, _, _ = _judge_quality(
        "Summarize this text",
        "Good summary here",
        _FailProvider(),
        fallback_factory=factory,
    )

    assert factory_calls == 1, f"Expected 1 factory call, got {factory_calls}"
    assert score == 0.70
    assert fb.call_count >= 1


@patch(_SLEEP_PATCH)
def test_fallback_factory_none_returns_unavailable(_sleep):
    """When fallback_factory returns None, judge returns 0.5 unavailable."""
    score, reason, _ = _judge_quality(
        "Summarize this text",
        "Good summary here",
        _FailProvider(),
        fallback_factory=lambda: None,
    )

    assert score == 0.5
    assert reason == "Judge unavailable"


@patch(_SLEEP_PATCH)
def test_no_fallback_factory_returns_unavailable(_sleep):
    """When no fallback_factory is provided, judge returns 0.5 unavailable."""
    score, reason, _ = _judge_quality(
        "Summarize this text",
        "Good summary here",
        _FailProvider(),
    )

    assert score == 0.5
    assert reason == "Judge unavailable"


def test_primary_regex_fallback_no_factory_call():
    """When primary returns non-JSON but regex-parseable, no factory call."""
    factory_called = False

    class _RegexProvider:
        model = "test-regex"
        def is_available(self): return True
        def complete(self, prompt, config=None):
            return _make_response('Here is my assessment. {"score": 0.92}')

    def factory():
        nonlocal factory_called
        factory_called = True
        return _FallbackProvider()

    score, _, _ = _judge_quality(
        "Summarize this text",
        "Good summary here",
        _RegexProvider(),
        fallback_factory=factory,
    )

    assert score == 0.92
    assert not factory_called, "Fallback factory called despite regex success"


# ---------------------------------------------------------------------------
# Integration: LiveEvaluator does NOT probe ollama when remote judge works
# ---------------------------------------------------------------------------


def test_pick_judge_no_ollama_probe_when_remote_available():
    """_pick_judge should not access ollama when a remote API provider works."""
    ollama_accessed = False

    def _trap_ollama(self):
        nonlocal ollama_accessed
        ollama_accessed = True
        return None

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False

    with patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(_trap_ollama)):
        judge = evaluator._pick_judge("gemini")

    assert judge is evaluator.gemini
    assert not ollama_accessed, "ollama accessed despite remote judge available"


def test_evaluate_task_no_ollama_probe_on_success():
    """Full evaluate_task() path should not resolve ollama when remote judge works.

    Mocks the organism execution and provider construction so evaluate_task()
    runs end-to-end, exercises the real judge call site, and verifies ollama
    is never touched.
    """
    from unittest.mock import MagicMock
    from operon_ai.patterns.types import SkillStageResult

    local_accessed = {}

    def _trap_ollama(self):
        local_accessed["ollama"] = True
        return None

    def _trap_lmstudio(self):
        local_accessed["lmstudio"] = True
        return None

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False

    mock_stage_result = SkillStageResult(
        stage_name="reader_0", role="reader", output={"summary": "test"},
        model_alias="fast", provider="mock", model="mock",
        tokens_used=50, latency_ms=100.0, action_type="EXECUTE",
        metadata={},
    )
    mock_run_result = MagicMock()
    mock_run_result.stage_results = (mock_stage_result,)
    mock_run_result.final_output = "A good summary of the JWST findings."

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(_trap_ollama)),
        patch.object(LiveEvaluator, 'lmstudio', new_callable=lambda: property(_trap_lmstudio)),
        patch("eval.convergence.live_evaluator.skill_organism") as mock_so,
    ):
        mock_organism = MagicMock()
        mock_organism.stages = [MagicMock(name="reader_0", role="reader", mode="fuzzy")]
        mock_organism.run.return_value = mock_run_result
        mock_so.return_value = mock_organism

        from eval.convergence.tasks import get_benchmark_tasks
        task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

        result = evaluator.evaluate_task(task, guided=False, provider_name="gemini")

    assert result.quality_score == 0.85, f"Expected 0.85, got {result.quality_score}"
    assert "ollama" not in local_accessed, "ollama accessed during evaluate_task"
    assert "lmstudio" not in local_accessed, "lmstudio accessed during evaluate_task"


def test_evaluate_cli_no_local_probe_on_success():
    """Full _evaluate_cli() path should not resolve ollama or lmstudio when remote judge works.

    Mocks cli_handler so no real subprocess runs, and traps both local properties.
    """
    from operon_ai.patterns.cli import CLIResult

    local_accessed = {}

    def _trap_ollama(self):
        local_accessed["ollama"] = True
        return None

    def _trap_lmstudio(self):
        local_accessed["lmstudio"] = True
        return None

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False

    mock_cli_result = CLIResult(
        stdout="JWST found CO2 on WASP-39b.",
        stderr="",
        returncode=0,
        command="claude -p",
        latency_ms=500.0,
    )

    def fake_handler(task_str):
        return {
            "output": mock_cli_result.stdout,
            "cli_result": mock_cli_result,
            "_action_type": "EXECUTE",
        }

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(_trap_ollama)),
        patch.object(LiveEvaluator, 'lmstudio', new_callable=lambda: property(_trap_lmstudio)),
        patch("eval.convergence.live_evaluator.cli_handler", return_value=fake_handler),
    ):
        result = evaluator.evaluate_task(task, guided=False, provider_name="claude")

    assert result.quality_score == 0.85, f"Expected 0.85, got {result.quality_score}"
    assert "ollama" not in local_accessed, "ollama accessed during _evaluate_cli"
    assert "lmstudio" not in local_accessed, "lmstudio accessed during _evaluate_cli"


# ---------------------------------------------------------------------------
# Cross-judging
# ---------------------------------------------------------------------------


def test_cross_judge_uses_specified_provider():
    """_pick_judge with judge_provider='lmstudio' returns the LM Studio provider."""
    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator._lmstudio = _FallbackProvider()  # Simulate detected LM Studio

    judge = evaluator._pick_judge("gemini", judge_provider="lmstudio")

    assert judge is evaluator._lmstudio, "Expected LM Studio provider, got something else"


def test_cross_judge_unavailable_raises():
    """_pick_judge raises ValueError when requested judge is not available."""
    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator._lmstudio = None  # LM Studio not available

    try:
        evaluator._pick_judge("gemini", judge_provider="lmstudio")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "lmstudio" in str(e).lower()


def test_default_judge_is_self_judge():
    """_pick_judge with judge_provider=None preserves self-judging behavior."""
    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False

    judge = evaluator._pick_judge("gemini", judge_provider=None)

    assert judge is evaluator.gemini, "Default should self-judge with execution provider"


def test_cross_judge_result_records_judge_provider():
    """evaluate_task records judge_provider in LiveRunResult."""
    from unittest.mock import MagicMock
    from operon_ai.patterns.types import SkillStageResult

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False
    evaluator._lmstudio = _SuccessProvider()
    evaluator._lmstudio.model = "nvidia/nemotron-3-nano-4b"

    mock_stage_result = SkillStageResult(
        stage_name="reader_0", role="reader", output="done",
        model_alias="fast", provider="mock", model="mock",
        tokens_used=10, latency_ms=50.0, action_type="EXECUTE",
        metadata={},
    )
    mock_run_result = MagicMock()
    mock_run_result.stage_results = (mock_stage_result,)
    mock_run_result.final_output = "Summary output."

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: None)),
        patch("eval.convergence.live_evaluator.skill_organism") as mock_so,
    ):
        mock_organism = MagicMock()
        mock_organism.stages = [MagicMock(name="reader_0", role="reader", mode="fuzzy")]
        mock_organism.run.return_value = mock_run_result
        mock_so.return_value = mock_organism

        result = evaluator.evaluate_task(
            task, guided=False, provider_name="gemini", judge_provider="lmstudio",
        )

    assert result.judge_provider == "lmstudio", f"Expected 'lmstudio', got {result.judge_provider!r}"


def test_invalid_judge_raises_before_execution():
    """evaluate_task raises ValueError for bad judge_provider before running the task."""
    from unittest.mock import MagicMock

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator._lmstudio = None  # Not available

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    try:
        with patch("eval.convergence.live_evaluator.skill_organism") as mock_so:
            mock_so.side_effect = AssertionError("should not reach execution")
            evaluator.evaluate_task(
                task, guided=False, provider_name="gemini", judge_provider="lmstudio",
            )
        assert False, "Expected ValueError to propagate"
    except ValueError as e:
        assert "lmstudio" in str(e).lower()


@patch(_SLEEP_PATCH)
def test_cross_judge_no_fallback(_sleep):
    """When judge_provider is explicitly set, fallback_factory should be None."""
    captured_fb = []

    original_judge = _judge_quality.__wrapped__ if hasattr(_judge_quality, '__wrapped__') else None

    # Patch _judge_quality to capture fallback_factory arg
    def _capture_judge(prompt, output, provider, *, fallback_factory=None):
        captured_fb.append(fallback_factory)
        return 0.85, ""

    from unittest.mock import MagicMock
    from operon_ai.patterns.types import SkillStageResult

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False
    evaluator._lmstudio = _SuccessProvider()

    mock_stage_result = SkillStageResult(
        stage_name="reader_0", role="reader", output="done",
        model_alias="fast", provider="mock", model="mock",
        tokens_used=10, latency_ms=50.0, action_type="EXECUTE",
        metadata={},
    )
    mock_run_result = MagicMock()
    mock_run_result.stage_results = (mock_stage_result,)
    mock_run_result.final_output = "Summary output."

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: None)),
        patch("eval.convergence.live_evaluator.skill_organism") as mock_so,
        patch("eval.convergence.live_evaluator._judge_quality", _capture_judge),
    ):
        mock_organism = MagicMock()
        mock_organism.stages = [MagicMock(name="reader_0", role="reader", mode="fuzzy")]
        mock_organism.run.return_value = mock_run_result
        mock_so.return_value = mock_organism

        evaluator.evaluate_task(
            task, guided=False, provider_name="gemini", judge_provider="lmstudio",
        )

    assert len(captured_fb) == 1, f"Expected 1 judge call, got {len(captured_fb)}"
    assert captured_fb[0] is None, "fallback_factory should be None when judge_provider is set"


def test_self_judge_records_actual_judge_for_cli():
    """Default self-judging for CLI providers should record the actual judge, not 'claude'."""
    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.gemini.name = "gemini"
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False

    # _pick_judge("claude") should return gemini (first available), not "claude"
    judge = evaluator._pick_judge("claude")
    assert judge is evaluator.gemini


def test_cli_cross_judge_result_metadata():
    """evaluate_task with CLI provider and cross-judge records correct judge_provider."""
    from operon_ai.patterns.cli import CLIResult

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.gemini.name = "gemini"
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False
    # Set raw field — don't patch property, cross-judging needs it
    evaluator._lmstudio = _SuccessProvider()

    mock_cli_result = CLIResult(
        stdout="JWST found CO2.", stderr="", returncode=0,
        command="claude -p", latency_ms=500.0,
    )

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: None)),
        patch("eval.convergence.live_evaluator.cli_handler", return_value=lambda task_str: {
            "output": mock_cli_result.stdout,
            "cli_result": mock_cli_result,
            "_action_type": "EXECUTE",
        }),
    ):
        result = evaluator.evaluate_task(
            task, guided=False, provider_name="claude", judge_provider="lmstudio",
        )

    assert result.judge_provider == "lmstudio", f"Expected 'lmstudio', got {result.judge_provider!r}"


@patch(_SLEEP_PATCH)
def test_cli_cross_judge_no_fallback(_sleep):
    """CLI cross-judging should disable fallback_factory."""
    captured_fb = []

    def _capture_judge(prompt, output, provider, *, fallback_factory=None):
        captured_fb.append(fallback_factory)
        return 0.85, ""

    from operon_ai.patterns.cli import CLIResult

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False
    evaluator._lmstudio = _SuccessProvider()

    mock_cli_result = CLIResult(
        stdout="JWST found CO2.", stderr="", returncode=0,
        command="claude -p", latency_ms=500.0,
    )

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: None)),
        patch("eval.convergence.live_evaluator.cli_handler", return_value=lambda task_str: {
            "output": mock_cli_result.stdout,
            "cli_result": mock_cli_result,
            "_action_type": "EXECUTE",
        }),
        patch("eval.convergence.live_evaluator._judge_quality", _capture_judge),
    ):
        evaluator.evaluate_task(
            task, guided=False, provider_name="claude", judge_provider="lmstudio",
        )

    assert len(captured_fb) == 1
    assert captured_fb[0] is None, "CLI cross-judge should have fallback_factory=None"


@patch(_SLEEP_PATCH)
def test_cli_self_judge_keeps_fallback(_sleep):
    """CLI self-judging (judge_provider=None) must keep Ollama fallback that resolves correctly."""
    captured_fb = []

    def _capture_judge(prompt, output, provider, *, fallback_factory=None):
        captured_fb.append(fallback_factory)
        return 0.85, ""

    from operon_ai.patterns.cli import CLIResult

    ollama_sentinel = _FallbackProvider()

    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False

    mock_cli_result = CLIResult(
        stdout="JWST found CO2.", stderr="", returncode=0,
        command="claude -p", latency_ms=500.0,
    )

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: ollama_sentinel)),
        patch.object(LiveEvaluator, 'lmstudio', new_callable=lambda: property(lambda s: None)),
        patch("eval.convergence.live_evaluator.cli_handler", return_value=lambda task_str: {
            "output": mock_cli_result.stdout,
            "cli_result": mock_cli_result,
            "_action_type": "EXECUTE",
        }),
        patch("eval.convergence.live_evaluator._judge_quality", _capture_judge),
    ):
        evaluator.evaluate_task(
            task, guided=False, provider_name="claude",
        )

        # Invoke factory inside patch scope (factory closes over self.ollama)
        assert len(captured_fb) == 1
        factory = captured_fb[0]
        assert factory is not None, "CLI self-judge should have fallback enabled"
        assert callable(factory), "fallback_factory must be callable"
        resolved = factory()
        assert resolved is ollama_sentinel, f"Factory should resolve to Ollama, got {type(resolved).__name__}"


def test_failed_run_preserves_judge_metadata():
    """When execution fails, LiveRunResult should still record judge_provider."""
    evaluator = LiveEvaluator()
    evaluator.gemini = _SuccessProvider()
    evaluator.anthropic = _FailProvider()
    evaluator.anthropic.is_available = lambda: False
    evaluator.openai = _FailProvider()
    evaluator.openai.is_available = lambda: False
    evaluator._lmstudio = _SuccessProvider()

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    from unittest.mock import MagicMock

    with (
        patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: None)),
        patch("eval.convergence.live_evaluator.skill_organism") as mock_so,
    ):
        mock_organism = MagicMock()
        mock_organism.run.side_effect = RuntimeError("boom")
        mock_so.return_value = mock_organism

        result = evaluator.evaluate_task(
            task, guided=False, provider_name="gemini", judge_provider="lmstudio",
        )

    assert result.success is False
    assert result.judge_provider == "lmstudio", f"Expected 'lmstudio' on failure path, got {result.judge_provider!r}"


# ---------------------------------------------------------------------------
# Regression: Gemini model IDs in evaluate_task
# ---------------------------------------------------------------------------


def test_gemini_model_ids_guided_and_unguided():
    """Verify exact Gemini model IDs for both guided and unguided configs."""
    from unittest.mock import MagicMock, call
    from operon_ai.patterns.types import SkillStageResult

    captured: dict[str, dict] = {"guided": {}, "unguided": {}}

    for is_guided in [False, True]:
        label = "guided" if is_guided else "unguided"
        models_used: list[str] = []

        def _capture_init(self, *args, model="gemini-2.5-flash", **kwargs):
            models_used.append(model)
            self.model = model
            self._api_key = "fake"
            self._client = None

        evaluator = LiveEvaluator()
        evaluator.gemini = _SuccessProvider()
        evaluator.anthropic = _FailProvider()
        evaluator.anthropic.is_available = lambda: False
        evaluator.openai = _FailProvider()
        evaluator.openai.is_available = lambda: False

        mock_stage_result = SkillStageResult(
            stage_name="reader_0", role="reader", output="done",
            model_alias="fast", provider="mock", model="mock",
            tokens_used=10, latency_ms=50.0, action_type="EXECUTE",
            metadata={},
        )
        mock_run_result = MagicMock()
        mock_run_result.stage_results = (mock_stage_result,)
        mock_run_result.final_output = "Summary output."

        from eval.convergence.tasks import get_benchmark_tasks
        task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

        with (
            patch.object(LiveEvaluator, 'ollama', new_callable=lambda: property(lambda s: None)),
            patch.object(LiveEvaluator, 'lmstudio', new_callable=lambda: property(lambda s: None)),
            patch("eval.convergence.live_evaluator.GeminiProvider.__init__", _capture_init),
            patch("eval.convergence.live_evaluator.skill_organism") as mock_so,
        ):
            mock_organism = MagicMock()
            mock_organism.stages = [MagicMock(name="reader_0", role="reader", mode="fuzzy")]
            mock_organism.run.return_value = mock_run_result
            mock_so.return_value = mock_organism

            evaluator.evaluate_task(task, guided=is_guided, provider_name="gemini")

            # Capture the stages passed to skill_organism
            stages_arg = mock_so.call_args[1].get("stages", mock_so.call_args[0][0] if mock_so.call_args[0] else [])

        captured[label] = {
            "models": list(models_used),
            "stage_modes": [s.mode for s in stages_arg],
            "stage_roles": [s.role for s in stages_arg],
        }

    # easy_seq_01 has required_roles=("reader", "writer") — exactly 2 stages
    # Model selection
    assert captured["unguided"]["models"] == ["gemini-2.5-flash", "gemini-2.5-flash"], \
        f"Unguided models: {captured['unguided']['models']}"
    assert captured["guided"]["models"] == ["gemini-2.5-flash", "gemini-2.5-pro"], \
        f"Guided models: {captured['guided']['models']}"

    # Stage roles match task.required_roles in order
    assert captured["unguided"]["stage_roles"] == ["reader", "writer"], \
        f"Unguided roles: {captured['unguided']['stage_roles']}"
    assert captured["guided"]["stage_roles"] == ["reader", "writer"], \
        f"Guided roles: {captured['guided']['stage_roles']}"

    # Stage modes: exact sequences
    assert captured["unguided"]["stage_modes"] == ["fuzzy", "fuzzy"], \
        f"Unguided modes: {captured['unguided']['stage_modes']}"
    assert captured["guided"]["stage_modes"] == ["fuzzy", "fixed"], \
        f"Guided modes: {captured['guided']['stage_modes']}"


def test_guided_nucleus_wiring_provider_assignment():
    """Guided configs assign fast_nucleus=cheap model (fixed stages), deep_nucleus=best model (fuzzy stages)."""
    from unittest.mock import MagicMock
    from operon_ai.patterns.types import SkillStageResult

    mock_stage_result = SkillStageResult(
        stage_name="reader_0", role="reader", output="done",
        model_alias="fast", provider="mock", model="mock",
        tokens_used=10, latency_ms=50.0, action_type="EXECUTE",
        metadata={},
    )
    mock_run_result = MagicMock()
    mock_run_result.stage_results = (mock_stage_result,)
    mock_run_result.final_output = "Summary output."

    from eval.convergence.tasks import get_benchmark_tasks
    task = next(t for t in get_benchmark_tasks() if t.task_id == "easy_seq_01")

    # Test all three API providers, guided and unguided
    expected = {
        "gemini": {
            "guided": ("gemini-2.5-flash", "gemini-2.5-pro"),
            "unguided": ("gemini-2.5-flash", "gemini-2.5-flash"),
        },
        "openai": {
            "guided": ("gpt-5.4-mini", "gpt-5.4"),
            "unguided": ("gpt-5.4-mini", "gpt-5.4-mini"),
        },
        "anthropic": {
            "guided": ("claude-haiku-4-5-20251001", "claude-sonnet-4-6-20260301"),
            "unguided": ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
        },
    }

    for provider_name, configs in expected.items():
        for is_guided, (exp_fast, exp_deep) in [(True, configs["guided"]), (False, configs["unguided"])]:
            label = f"{provider_name}/{'guided' if is_guided else 'unguided'}"
            evaluator = LiveEvaluator()
            evaluator.gemini = _SuccessProvider()
            evaluator.anthropic = _FailProvider()
            evaluator.anthropic.is_available = lambda: False
            evaluator.openai = _FailProvider()
            evaluator.openai.is_available = lambda: False

            with (
                patch.object(LiveEvaluator, '_pick_judge', return_value=_SuccessProvider()),
                patch("eval.convergence.live_evaluator.skill_organism") as mock_so,
                patch("eval.convergence.live_evaluator._judge_quality", return_value=(0.85, "", _SuccessProvider())),
            ):
                mock_organism = MagicMock()
                mock_organism.stages = [MagicMock(name="reader_0", role="reader", mode="fuzzy")]
                mock_organism.run.return_value = mock_run_result
                mock_so.return_value = mock_organism

                evaluator.evaluate_task(task, guided=is_guided, provider_name=provider_name)

                call_kwargs = mock_so.call_args[1]
                fast_model = call_kwargs["fast_nucleus"].provider.model
                deep_model = call_kwargs["deep_nucleus"].provider.model

            assert fast_model == exp_fast, \
                f"{label}: fast_nucleus should be {exp_fast}, got {fast_model}"
            assert deep_model == exp_deep, \
                f"{label}: deep_nucleus should be {exp_deep}, got {deep_model}"
