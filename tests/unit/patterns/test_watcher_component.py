from dataclasses import dataclass
from typing import Any

from operon_ai import (
    InterventionKind,
    SkillStage,
    WatcherComponent,
    WatcherConfig,
    WatcherIntervention,
)
from operon_ai.patterns.watcher import SignalCategory, WATCHER_STATE_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    output: Any = "some output"
    model_alias: str = "fast"
    action_type: str = "EXECUTE"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class _FakeEpiplexityResult:
    epiplexity: float = 0.5
    epiplexic_integral: float = 0.5
    status: Any = None

    def __post_init__(self):
        if self.status is None:
            from enum import Enum
            _S = Enum("_S", {"HEALTHY": "healthy"})
            self.status = _S.HEALTHY


class _FakeEpiplexityMonitor:
    def __init__(self, status_value="healthy", epiplexity=0.5):
        self._status_value = status_value
        self._epiplexity = epiplexity

    def measure(self, message, perplexity=None):
        from enum import Enum
        _S = Enum("_S", {
            "HEALTHY": "healthy",
            "STAGNANT": "stagnant",
            "CRITICAL": "critical",
            "CONVERGING": "converging",
            "EXPLORING": "exploring",
        })
        return _FakeEpiplexityResult(
            epiplexity=self._epiplexity,
            status=_S[self._status_value.upper()],
        )


class _FakeImmuneResponse:
    def __init__(self, threat_level_value, action_value="ignore"):
        from enum import Enum
        _TL = Enum("_TL", {"NONE": "none", "SUSPICIOUS": "suspicious", "CONFIRMED": "confirmed", "CRITICAL": "critical"})
        _A = Enum("_A", {"IGNORE": "ignore", "QUARANTINE": "quarantine"})
        self.threat_level = _TL[threat_level_value.upper()]
        self.action = _A[action_value.upper()]


class _FakeImmuneSystem:
    def __init__(self, threat_level="none"):
        self._threat_level = threat_level

    def inspect(self, agent_id):
        return _FakeImmuneResponse(self._threat_level)


class _FakeBudget:
    def __init__(self, atp=100, max_atp=100):
        self.atp = atp
        self.max_atp = max_atp


def _stage(name="s1"):
    return SkillStage(name=name, role="Worker")


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_watcher_satisfies_runtime_component_protocol():
    w = WatcherComponent()
    assert hasattr(w, "on_run_start")
    assert hasattr(w, "on_stage_start")
    assert hasattr(w, "on_stage_result")
    assert hasattr(w, "on_run_complete")


def test_watcher_with_no_signal_sources_is_noop():
    w = WatcherComponent()
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_start(_stage(), state, {})
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    w.on_run_complete(_FakeResult(), state)
    assert WATCHER_STATE_KEY not in state
    assert w.summary()["total_interventions"] == 0


# ---------------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------------


def test_epistemic_signal_from_epiplexity_monitor():
    monitor = _FakeEpiplexityMonitor(status_value="healthy", epiplexity=0.6)
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    assert w.summary()["signals_by_category"].get("epistemic", 0) >= 1


def test_somatic_signal_from_atp_store():
    budget = _FakeBudget(atp=50, max_atp=100)
    w = WatcherComponent(budget=budget)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_start(_stage(), state, {})
    assert w.summary()["signals_by_category"].get("somatic", 0) >= 1


def test_species_signal_from_immune_system():
    immune = _FakeImmuneSystem(threat_level="none")
    w = WatcherComponent(immune_system=immune, immune_agent_id="s1")
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    assert w.summary()["signals_by_category"].get("species", 0) >= 1


def test_signals_accumulated_across_stages():
    monitor = _FakeEpiplexityMonitor()
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage("s1"), _FakeResult(), state, {})
    w.on_stage_result(_stage("s2"), _FakeResult(), state, {})
    assert w.summary()["total_stages_observed"] == 2
    assert w.summary()["total_signals"] >= 2


# ---------------------------------------------------------------------------
# Intervention decisions
# ---------------------------------------------------------------------------


def test_no_intervention_on_healthy_signals():
    monitor = _FakeEpiplexityMonitor(status_value="healthy")
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    assert WATCHER_STATE_KEY not in state
    assert w.summary()["total_interventions"] == 0


def test_retry_on_failure_result():
    w = WatcherComponent()
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(action_type="FAILURE"), state, {})
    assert WATCHER_STATE_KEY in state
    intervention = state[WATCHER_STATE_KEY]
    assert isinstance(intervention, WatcherIntervention)
    assert intervention.kind == InterventionKind.RETRY


def test_escalate_on_stagnant_epiplexity():
    monitor = _FakeEpiplexityMonitor(status_value="stagnant")
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(model_alias="fast"), state, {})
    assert WATCHER_STATE_KEY in state
    assert state[WATCHER_STATE_KEY].kind == InterventionKind.ESCALATE


def test_escalate_on_critical_epiplexity():
    monitor = _FakeEpiplexityMonitor(status_value="critical")
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(model_alias="fast"), state, {})
    assert state[WATCHER_STATE_KEY].kind == InterventionKind.ESCALATE


def test_halt_on_critical_immune_threat():
    immune = _FakeImmuneSystem(threat_level="critical")
    w = WatcherComponent(immune_system=immune, immune_agent_id="s1")
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    assert WATCHER_STATE_KEY in state
    assert state[WATCHER_STATE_KEY].kind == InterventionKind.HALT


def test_halt_on_exceeded_intervention_rate():
    """When intervention rate exceeds threshold, HALT for non-convergence."""
    w = WatcherComponent(config=WatcherConfig(max_intervention_rate=0.3))
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    # Manually inject interventions to exceed rate
    w.interventions.append(WatcherIntervention(InterventionKind.RETRY, "s0", "test"))
    w._total_stages = 1  # rate = 1/1 = 1.0 > 0.3
    w.on_stage_result(_stage("s1"), _FakeResult(), state, {})
    assert WATCHER_STATE_KEY in state
    assert state[WATCHER_STATE_KEY].kind == InterventionKind.HALT
    assert "non-convergence" in state[WATCHER_STATE_KEY].reason


def test_retry_count_respects_max_retries_per_stage():
    w = WatcherComponent(config=WatcherConfig(max_retries_per_stage=1))
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    # First failure → RETRY
    w.on_stage_result(_stage("s1"), _FakeResult(action_type="FAILURE"), state, {})
    assert state.pop(WATCHER_STATE_KEY).kind == InterventionKind.RETRY
    # Second failure → no more retries
    w.on_stage_result(_stage("s1"), _FakeResult(action_type="FAILURE"), state, {})
    assert WATCHER_STATE_KEY not in state


# ---------------------------------------------------------------------------
# shared_state interaction
# ---------------------------------------------------------------------------


def test_intervention_written_to_shared_state():
    w = WatcherComponent()
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(action_type="FAILURE"), state, {})
    assert WATCHER_STATE_KEY in state


def test_intervention_key_uses_config_state_key():
    custom_key = "_my_watcher"
    w = WatcherComponent(config=WatcherConfig(state_key=custom_key))
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(action_type="FAILURE"), state, {})
    assert custom_key in state


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_includes_signal_and_intervention_counts():
    monitor = _FakeEpiplexityMonitor(status_value="stagnant")
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(model_alias="fast"), state, {})
    s = w.summary()
    assert s["total_signals"] >= 1
    assert s["total_interventions"] >= 1
    assert "escalate" in s["interventions_by_kind"]
    assert s["total_stages_observed"] == 1
