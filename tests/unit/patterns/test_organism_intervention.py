"""Integration tests: WatcherComponent + SkillOrganism run loop interventions."""

from dataclasses import dataclass, field
from typing import Any

from operon_ai import (
    InterventionKind,
    SkillStage,
    WatcherComponent,
    WatcherConfig,
    WatcherIntervention,
    skill_organism,
)
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider
from operon_ai.patterns.types import WATCHER_STATE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_organism(stages, components=(), fast_responses=None, deep_responses=None):
    fast = Nucleus(provider=MockProvider(responses=fast_responses or {}))
    deep = Nucleus(provider=MockProvider(responses=deep_responses or {}))
    return skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=list(components),
    )


# A minimal watcher that forces a specific intervention on a target stage
@dataclass
class _ForcedWatcher:
    """Injects a predetermined intervention for a target stage."""
    target_stage: str
    intervention_kind: InterventionKind
    fired: bool = field(default=False, init=False)

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        pass

    def on_stage_start(self, stage: Any, shared_state: dict[str, Any], stage_outputs: dict[str, Any]) -> None:
        pass

    def on_stage_result(self, stage: Any, result: Any, shared_state: dict[str, Any], stage_outputs: dict[str, Any]) -> None:
        if not self.fired and getattr(stage, "name", "") == self.target_stage:
            self.fired = True
            shared_state[WATCHER_STATE_KEY] = WatcherIntervention(
                kind=self.intervention_kind,
                stage_name=self.target_stage,
                reason="forced by test",
            )

    def on_run_complete(self, result: Any, shared_state: dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# RETRY
# ---------------------------------------------------------------------------

_call_count = 0

def test_organism_retries_stage_on_watcher_retry_intervention():
    global _call_count
    _call_count = 0

    def flaky_handler(task):
        global _call_count
        _call_count += 1
        if _call_count == 1:
            return "first attempt"
        return "retry attempt"

    watcher = _ForcedWatcher(target_stage="flaky", intervention_kind=InterventionKind.RETRY)
    organism = _make_organism(
        stages=[
            SkillStage(name="flaky", role="Worker", handler=flaky_handler),
        ],
        components=[watcher],
    )
    result = organism.run("test")
    assert result.final_output == "retry attempt"
    assert _call_count == 2


# ---------------------------------------------------------------------------
# ESCALATE
# ---------------------------------------------------------------------------


def test_organism_escalates_to_deep_on_watcher_escalate_intervention():
    watcher = _ForcedWatcher(target_stage="router", intervention_kind=InterventionKind.ESCALATE)
    fast = Nucleus(provider=MockProvider(responses={
        "return a routing label": "fast response",
    }))
    deep = Nucleus(provider=MockProvider(responses={
        "return a routing label": "deep response",
    }))
    organism = skill_organism(
        stages=[
            SkillStage(name="router", role="Router", instructions="Return a routing label.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[watcher],
    )
    result = organism.run("test")
    # After escalation, the deep nucleus should have produced the result
    assert result.final_output == "deep response"


# ---------------------------------------------------------------------------
# HALT
# ---------------------------------------------------------------------------


def test_organism_halts_on_watcher_halt_intervention():
    watcher = _ForcedWatcher(target_stage="s1", intervention_kind=InterventionKind.HALT)
    organism = _make_organism(
        stages=[
            SkillStage(name="s1", role="First", handler=lambda task: "one"),
            SkillStage(name="s2", role="Second", handler=lambda task: "two"),
        ],
        components=[watcher],
    )
    result = organism.run("test")
    # Should halt after s1 — s2 never runs
    assert len(result.stage_results) == 1
    assert result.final_output == "one"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_organism_runs_normally_without_watcher():
    organism = _make_organism(
        stages=[
            SkillStage(name="echo", role="Echo", handler=lambda task: task),
        ],
    )
    result = organism.run("hello")
    assert result.final_output == "hello"


def test_intervention_key_not_in_final_shared_state():
    watcher = _ForcedWatcher(target_stage="s1", intervention_kind=InterventionKind.RETRY)
    organism = _make_organism(
        stages=[
            SkillStage(name="s1", role="Worker", handler=lambda task: "result"),
        ],
        components=[watcher],
    )
    result = organism.run("test")
    assert WATCHER_STATE_KEY not in result.shared_state
