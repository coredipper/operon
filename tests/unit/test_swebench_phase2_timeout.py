"""Tests for SWE-bench Phase 2 LLM-call timeout configuration.

Context: the original 2026-04-17 grounded rerun had two baseline calls
time out at the OpenAI-compatible client's default (ProviderConfig
.timeout_seconds = 120s). Reasoning models like deepseek-r1 routinely
produce 5-10 minute responses (long <think> blocks before the final
answer), so 120s is not a safe default for SWE-bench-style prompts
with or without grounding. This test locks in a timeout that accommodates
both current gemma4 runs (mean 131s, some instances higher) and future
reasoning-model reruns without retroactive configuration.

The fix is structural, not model-specific: the default belongs in the
swebench_phase2 runner because the regular ProviderConfig default
(120s) is correct for small interactive callers and would be a
regression to change globally.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.swebench_phase2 import _LLM_TIMEOUT_SECONDS, _build_organism, _llm_call  # noqa: E402
from operon_ai.providers.base import LLMResponse, ProviderConfig  # noqa: E402


def test_llm_timeout_default_is_at_least_900_seconds():
    """The module-level default must be large enough to accommodate
    reasoning models (5-10 min think + generate) and large-context
    grounded prompts on mid-tier 8B models. 900s is a safety ceiling,
    not a tight fit. Review follow-up to v0.34.5 Phase C readiness."""
    assert _LLM_TIMEOUT_SECONDS >= 900.0, (
        f"_LLM_TIMEOUT_SECONDS={_LLM_TIMEOUT_SECONDS} is too short for "
        f"reasoning models; bump to 900s minimum"
    )


class _CaptureProvider:
    """Test double that records the ProviderConfig passed to complete().

    Avoids any network / Ollama interaction while letting us assert on
    the exact timeout value the real provider would have been invoked
    with. We record config rather than replaying it because the test's
    concern is the contract between _llm_call and the provider, not
    what the provider then does with it.
    """

    def __init__(self):
        self.last_config: ProviderConfig | None = None
        self.last_prompt: str | None = None

    def complete(
        self, prompt: str, config: ProviderConfig | None = None
    ) -> LLMResponse:
        self.last_prompt = prompt
        self.last_config = config
        return LLMResponse(
            content="", model="test", tokens_used=0, latency_ms=0.0,
            raw_response={},
        )


def test_llm_call_forwards_module_default_timeout_to_provider():
    """_llm_call must populate ProviderConfig.timeout_seconds with the
    module default, not leave the provider's own default (120s) in
    place. Without this, reasoning-model reruns hit the 120s ceiling."""
    provider = _CaptureProvider()

    _llm_call(provider, "any prompt")

    assert provider.last_config is not None
    assert provider.last_config.timeout_seconds == _LLM_TIMEOUT_SECONDS


def test_llm_call_preserves_existing_max_tokens_setting():
    """Sanity: bumping the timeout must not disturb the pre-existing
    max_tokens=4096 the runner relies on for full-diff outputs."""
    provider = _CaptureProvider()

    _llm_call(provider, "any prompt")

    assert provider.last_config is not None
    assert provider.last_config.max_tokens == 4096


def test_llm_probe_timeout_is_short_enough_to_fail_fast():
    """The reachability probe must fail within roughly a connect+load
    budget (Ollama cold-loads can take a minute on large quants).
    The benchmark default (900s) is a regression for a healthcheck —
    if the model is misconfigured, we want the error in ≤2 min, not
    ≤15. Review #753 from v0.34.5 follow-up."""
    from eval.swebench_phase2 import _LLM_PROBE_TIMEOUT_SECONDS  # noqa: PLC0415
    assert 10.0 <= _LLM_PROBE_TIMEOUT_SECONDS <= 120.0, (
        f"_LLM_PROBE_TIMEOUT_SECONDS={_LLM_PROBE_TIMEOUT_SECONDS} "
        f"must be a reachability budget (10-120s), not a benchmark "
        f"budget"
    )
    # Concrete contract: probe strictly shorter than benchmark.
    assert _LLM_PROBE_TIMEOUT_SECONDS < _LLM_TIMEOUT_SECONDS


def test_llm_call_accepts_timeout_override():
    """Callers must be able to pass a shorter timeout without mutating
    the module default. This is the mechanism the probe uses to avoid
    the 900s hang."""
    provider = _CaptureProvider()

    _llm_call(provider, "ping", timeout_seconds=30.0)

    assert provider.last_config is not None
    assert provider.last_config.timeout_seconds == 30.0


def test_llm_call_default_unchanged_when_override_not_passed():
    """Sanity: the new override parameter defaults to None (or the
    module default) so existing callers keep the 900s budget."""
    provider = _CaptureProvider()

    _llm_call(provider, "full benchmark prompt")

    assert provider.last_config is not None
    assert provider.last_config.timeout_seconds == _LLM_TIMEOUT_SECONDS


def test_probe_model_uses_probe_timeout():
    """The probe helper must route through the short timeout, not the
    benchmark default. Without this test, the probe could silently
    drift back to 900s if someone later edits _probe_model."""
    from eval.swebench_phase2 import (  # noqa: PLC0415
        _LLM_PROBE_TIMEOUT_SECONDS, _probe_model,
    )
    provider = _CaptureProvider()

    _probe_model(provider)

    assert provider.last_config is not None
    assert provider.last_config.timeout_seconds == _LLM_PROBE_TIMEOUT_SECONDS


def test_build_organism_stages_carry_llm_timeout():
    """Every SkillStage in the organism must ship with a
    provider_config whose timeout_seconds matches the module default.

    Without this, organism and langgraph runs (which go through
    BioAgent -> Nucleus.transcribe -> provider.complete with the stage's
    provider_config) silently fall back to the provider base's 120s
    default even though baseline is now correctly configured. Review
    #748 structural principle: when a value must be uniform across
    emission sites, delegate to a single source of truth, not a
    per-site replication."""
    organism = _build_organism(_CaptureProvider())

    for stage in organism.stages:
        assert stage.provider_config is not None, (
            f"stage '{stage.name}' has no provider_config; will hit "
            f"the 120s provider default and time out on reasoning models"
        )
        assert stage.provider_config.timeout_seconds == _LLM_TIMEOUT_SECONDS, (
            f"stage '{stage.name}' timeout_seconds="
            f"{stage.provider_config.timeout_seconds} does not match "
            f"module default {_LLM_TIMEOUT_SECONDS}"
        )
