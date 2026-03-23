"""Tests for curiosity signals in WatcherComponent."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from operon_ai import (
    InterventionKind,
    SkillStage,
    WatcherComponent,
    WatcherConfig,
)
from operon_ai.patterns.watcher import SignalCategory, WATCHER_STATE_KEY


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Status(Enum):
    HEALTHY = "healthy"
    EXPLORING = "exploring"
    STAGNANT = "stagnant"
    CRITICAL = "critical"


@dataclass
class _FakeEpiplexityResult:
    epiplexity: float = 0.5
    epiplexic_integral: float = 0.5
    embedding_novelty: float = 0.5
    status: Any = None

    def __post_init__(self):
        if self.status is None:
            self.status = _Status.HEALTHY


class _FakeEpiplexityMonitor:
    def __init__(self, status="healthy", novelty=0.5, epiplexity=0.5):
        self._status = status
        self._novelty = novelty
        self._epiplexity = epiplexity

    def measure(self, message, perplexity=None):
        return _FakeEpiplexityResult(
            epiplexity=self._epiplexity,
            epiplexic_integral=self._epiplexity,
            embedding_novelty=self._novelty,
            status=_Status[self._status.upper()],
        )


@dataclass
class _FakeResult:
    output: str = "some output"
    model_alias: str = "fast"
    action_type: str = "EXECUTE"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _stage(name="s1"):
    return SkillStage(name=name, role="Worker")


# ---------------------------------------------------------------------------
# Signal collection
# ---------------------------------------------------------------------------


def test_curiosity_signal_emitted_on_exploring():
    monitor = _FakeEpiplexityMonitor(status="exploring", novelty=0.8)
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    curiosity = [s for s in w.signals if s.source == "curiosity"]
    assert len(curiosity) == 1
    assert curiosity[0].value == 0.8


def test_no_curiosity_signal_on_healthy():
    monitor = _FakeEpiplexityMonitor(status="healthy", novelty=0.3)
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    curiosity = [s for s in w.signals if s.source == "curiosity"]
    assert len(curiosity) == 0


def test_no_curiosity_signal_on_stagnant():
    monitor = _FakeEpiplexityMonitor(status="stagnant", novelty=0.1)
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    curiosity = [s for s in w.signals if s.source == "curiosity"]
    assert len(curiosity) == 0


def test_no_curiosity_signal_without_monitor():
    w = WatcherComponent()
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    curiosity = [s for s in w.signals if s.source == "curiosity"]
    assert len(curiosity) == 0


def test_curiosity_signal_is_epistemic_category():
    monitor = _FakeEpiplexityMonitor(status="exploring", novelty=0.7)
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    curiosity = [s for s in w.signals if s.source == "curiosity"]
    assert curiosity[0].category == SignalCategory.EPISTEMIC


# ---------------------------------------------------------------------------
# Intervention chain
# ---------------------------------------------------------------------------


def test_curiosity_escalates_fast_model_above_threshold():
    monitor = _FakeEpiplexityMonitor(status="exploring", novelty=0.8)
    w = WatcherComponent(
        epiplexity_monitor=monitor,
        config=WatcherConfig(curiosity_escalation_threshold=0.5),
    )
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(model_alias="fast"), state, {})
    assert WATCHER_STATE_KEY in state
    assert state[WATCHER_STATE_KEY].kind == InterventionKind.ESCALATE
    assert "curiosity" in state[WATCHER_STATE_KEY].reason


def test_curiosity_no_escalate_on_deep_model():
    monitor = _FakeEpiplexityMonitor(status="exploring", novelty=0.9)
    w = WatcherComponent(
        epiplexity_monitor=monitor,
        config=WatcherConfig(curiosity_escalation_threshold=0.5),
    )
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(model_alias="deep"), state, {})
    assert WATCHER_STATE_KEY not in state


def test_curiosity_no_escalate_below_threshold():
    monitor = _FakeEpiplexityMonitor(status="exploring", novelty=0.3)
    w = WatcherComponent(
        epiplexity_monitor=monitor,
        config=WatcherConfig(curiosity_escalation_threshold=0.5),
    )
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(model_alias="fast"), state, {})
    assert WATCHER_STATE_KEY not in state


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_existing_epistemic_signals_still_emitted():
    monitor = _FakeEpiplexityMonitor(status="healthy", novelty=0.5, epiplexity=0.5)
    w = WatcherComponent(epiplexity_monitor=monitor)
    state: dict[str, Any] = {}
    w.on_run_start("task", state)
    w.on_stage_result(_stage(), _FakeResult(), state, {})
    epistemic = [s for s in w.signals if s.source == "epiplexity"]
    assert len(epistemic) >= 1


def test_default_curiosity_threshold():
    config = WatcherConfig()
    assert config.curiosity_escalation_threshold == 0.6
