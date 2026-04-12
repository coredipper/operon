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


# ---------------------------------------------------------------------------
# State leak across runs
# ---------------------------------------------------------------------------


def test_on_run_start_clears_signals_and_interventions():
    """Regression: signals and interventions must not leak across runs."""
    monitor = _FakeEpiplexityMonitor(status_value="stagnant")
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}

    # --- Run 1: accumulate signals and interventions ---
    w.on_run_start("task-1", state)
    w.on_stage_start(_stage("s1"), state, {})
    w.on_stage_result(_stage("s1"), _FakeResult(model_alias="fast"), state, {})
    w.on_run_complete(_FakeResult(), state)

    # Verify run 1 produced data
    assert len(w.signals) > 0, "Run 1 should have produced signals"
    assert len(w.interventions) > 0, "Run 1 should have produced interventions"

    # --- Run 2: on_run_start must clear prior-run state ---
    w.on_run_start("task-2", state)

    assert w.signals == [], "signals must be empty after on_run_start"
    assert w.interventions == [], "interventions must be empty after on_run_start"
    assert w._total_stages == 0
    assert w._stage_retry_counts == {}

    # summary and mode_balance should reflect the clean slate
    s = w.summary()
    assert s["total_signals"] == 0
    assert s["total_interventions"] == 0
    assert s["total_stages_observed"] == 0
    assert s["convergent"] is True

    mb = w.mode_balance()
    assert mb["observational"] == 0
    assert mb["action_oriented"] == 0
    assert mb["mismatches"] == 0


# ---------------------------------------------------------------------------
# RunContext typed accessors
# ---------------------------------------------------------------------------


def test_run_context_is_dict_subclass():
    """RunContext is a drop-in replacement for dict."""
    from operon_ai.patterns.types import RunContext
    ctx = RunContext({"key": "value"})
    assert isinstance(ctx, dict)
    assert ctx["key"] == "value"
    ctx["new"] = 42
    assert ctx.get("new") == 42


def test_run_context_watcher_intervention_none():
    from operon_ai.patterns.types import RunContext
    ctx = RunContext()
    assert ctx.watcher_intervention is None


def test_run_context_watcher_intervention_present():
    from operon_ai.patterns.types import (
        RunContext, InterventionKind, WatcherIntervention, WATCHER_STATE_KEY,
    )
    intervention = WatcherIntervention(
        kind=InterventionKind.HALT, stage_name="s1", reason="test",
    )
    ctx = RunContext({WATCHER_STATE_KEY: intervention})
    assert ctx.watcher_intervention is intervention


def test_run_context_verifier_signals_empty():
    from operon_ai.patterns.types import RunContext
    ctx = RunContext()
    assert ctx.verifier_signals == []


def test_run_context_verifier_signals_populated():
    from operon_ai.patterns.types import RunContext, _VERIFIER_SIGNALS_KEY
    ctx = RunContext({_VERIFIER_SIGNALS_KEY: ["sig1", "sig2"]})
    assert ctx.verifier_signals == ["sig1", "sig2"]


def test_run_context_telemetry_empty():
    from operon_ai.patterns.types import RunContext
    ctx = RunContext()
    assert ctx.telemetry_events == []


def test_run_context_used_by_organism():
    """SkillOrganism.run() wraps shared_state in RunContext."""
    from operon_ai.patterns.types import RunContext
    from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism

    org = skill_organism(
        stages=[SkillStage(name="s", role="r", instructions="do it")],
        fast_nucleus=Nucleus(provider=MockProvider(responses={"do it": "done"})),
        deep_nucleus=Nucleus(provider=MockProvider(responses={"do it": "done"})),
    )
    # Capture shared_state via a component
    captured = {}

    class Spy:
        def on_run_start(self, task, shared_state):
            captured["type"] = type(shared_state).__name__
        def on_stage_start(self, stage, shared_state, stage_outputs): pass
        def on_stage_result(self, stage, result, shared_state, stage_outputs): pass
        def on_run_complete(self, result, shared_state): pass

    org.components = (*org.components, Spy())  # type: ignore[assignment]
    org.run("test")
    assert captured["type"] == "RunContext"


def test_run_context_custom_watcher_key():
    """RunContext respects custom watcher key from WatcherConfig."""
    from operon_ai.patterns.types import (
        RunContext, InterventionKind, WatcherIntervention,
    )
    custom_key = "_my_watcher"
    intervention = WatcherIntervention(
        kind=InterventionKind.HALT, stage_name="s1", reason="test",
    )
    ctx = RunContext({custom_key: intervention}, watcher_key=custom_key)
    assert ctx.watcher_intervention is intervention

    # Default key should not find it
    ctx2 = RunContext({custom_key: intervention})
    assert ctx2.watcher_intervention is None


def test_run_context_append_persists():
    """Appending through verifier_signals accessor persists in the dict."""
    from operon_ai.patterns.types import RunContext, _VERIFIER_SIGNALS_KEY
    ctx = RunContext()
    ctx.verifier_signals.append("signal_1")
    ctx.verifier_signals.append("signal_2")
    assert ctx[_VERIFIER_SIGNALS_KEY] == ["signal_1", "signal_2"]
    assert ctx.verifier_signals == ["signal_1", "signal_2"]


def test_run_context_telemetry_append_persists():
    """Appending through telemetry_events accessor persists in the dict."""
    from operon_ai.patterns.types import RunContext, _TELEMETRY_KEY
    ctx = RunContext()
    ctx.telemetry_events.append("event_1")
    assert ctx[_TELEMETRY_KEY] == ["event_1"]


def test_organism_resolves_custom_watcher_key():
    """SkillOrganism.run() picks up WatcherConfig.state_key for RunContext."""
    from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
    from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig
    from operon_ai.patterns.types import RunContext

    custom_key = "_custom_intervention"
    watcher = WatcherComponent(config=WatcherConfig(state_key=custom_key))

    captured = {}

    class KeySpy:
        def on_run_start(self, task, shared_state):
            if isinstance(shared_state, RunContext):
                captured["watcher_key"] = shared_state._watcher_key
        def on_stage_start(self, stage, shared_state, stage_outputs): pass
        def on_stage_result(self, stage, result, shared_state, stage_outputs): pass
        def on_run_complete(self, result, shared_state): pass

    # HaltInjector writes a HALT under the custom key on the first stage,
    # so the second stage should never execute.
    from operon_ai.patterns.types import InterventionKind, WatcherIntervention

    class HaltInjector:
        def on_run_start(self, task, shared_state): pass
        def on_stage_start(self, stage, shared_state, stage_outputs):
            if getattr(stage, "name", "") == "s1":
                shared_state[custom_key] = WatcherIntervention(
                    kind=InterventionKind.HALT, stage_name="s1", reason="test halt",
                )
        def on_stage_result(self, stage, result, shared_state, stage_outputs): pass
        def on_run_complete(self, result, shared_state): pass

    org = skill_organism(
        stages=[
            SkillStage(name="s1", role="r1", instructions="do step 1"),
            SkillStage(name="s2", role="r2", instructions="do step 2"),
        ],
        fast_nucleus=Nucleus(provider=MockProvider(responses={"do step 1": "done1", "do step 2": "done2"})),
        deep_nucleus=Nucleus(provider=MockProvider(responses={"do step 1": "done1", "do step 2": "done2"})),
        components=[watcher, KeySpy(), HaltInjector()],
    )
    result = org.run("test")
    assert captured["watcher_key"] == custom_key
    # HALT on s1 should prevent s2 from executing
    executed_stages = [sr.stage_name for sr in result.stage_results]
    assert "s2" not in executed_stages, f"s2 should be halted but ran: {executed_stages}"


def test_certificate_gate_uses_custom_watcher_key():
    """CertificateGateComponent writes HALT to RunContext._watcher_key."""
    from operon_ai.patterns.certificate_gate import CertificateGateComponent
    from operon_ai.patterns.types import RunContext, WatcherIntervention
    from operon_ai.state.genome import Genome, Gene, GeneType
    from operon_ai.state.dna_repair import DNARepair

    genome = Genome(
        genes=[Gene(name="g1", gene_type=GeneType.REGULATORY, value="ok")],
        allow_mutations=True,
    )
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    # Corrupt the genome after checkpoint
    genome.mutate("g1", "corrupted")

    custom_key = "_my_gate_key"
    ctx = RunContext({}, watcher_key=custom_key)

    gate = CertificateGateComponent(
        genome=genome, repair=repair, checkpoint=checkpoint,
    )
    gate.on_run_start("test", ctx)

    class FakeStage:
        name = "s1"

    gate.on_stage_start(FakeStage(), ctx, {})

    # HALT should be written to the custom key, not the default
    from operon_ai.patterns.types import InterventionKind, WATCHER_STATE_KEY
    assert custom_key in ctx, f"Expected HALT under {custom_key}, got keys: {list(ctx.keys())}"
    assert isinstance(ctx[custom_key], WatcherIntervention)
    assert ctx[custom_key].kind == InterventionKind.HALT
    assert WATCHER_STATE_KEY not in ctx, "Default key should not be written when custom key is set"
