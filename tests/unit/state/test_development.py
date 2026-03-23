"""Tests for DevelopmentController and developmental staging."""

from operon_ai import (
    DevelopmentController,
    DevelopmentConfig,
    DevelopmentalStage,
    CriticalPeriod,
    Telomere,
)
from operon_ai.state.development import stage_reached, _STAGE_ORDER


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------


def test_stage_ordering():
    assert _STAGE_ORDER[DevelopmentalStage.EMBRYONIC] < _STAGE_ORDER[DevelopmentalStage.MATURE]


def test_stage_reached_same():
    assert stage_reached(DevelopmentalStage.JUVENILE, DevelopmentalStage.JUVENILE) is True


def test_stage_reached_higher():
    assert stage_reached(DevelopmentalStage.MATURE, DevelopmentalStage.JUVENILE) is True


def test_stage_not_reached_lower():
    assert stage_reached(DevelopmentalStage.EMBRYONIC, DevelopmentalStage.MATURE) is False


# ---------------------------------------------------------------------------
# DevelopmentController basics
# ---------------------------------------------------------------------------


def test_initial_stage_is_embryonic():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    assert dc.stage == DevelopmentalStage.EMBRYONIC


def test_tick_delegates_to_telomere():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    assert dc.tick() is True
    assert t._operations_count == 1


def test_tick_returns_false_when_depleted():
    t = Telomere(max_operations=5)
    t.start()
    dc = DevelopmentController(telomere=t)
    for _ in range(10):
        dc.tick()
    # Telomere should be depleted/senescent
    assert t._telomere_length == 0


def test_stage_transitions_at_juvenile():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t, config=DevelopmentConfig(juvenile_threshold=0.10))
    # Tick 11 times (11% consumed, past 10% threshold — avoids floating point edge)
    for _ in range(11):
        dc.tick()
    assert dc.stage == DevelopmentalStage.JUVENILE


def test_stage_transitions_at_mature():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t, config=DevelopmentConfig(
        juvenile_threshold=0.10, adolescent_threshold=0.35, mature_threshold=0.70,
    ))
    for _ in range(70):
        dc.tick()
    assert dc.stage == DevelopmentalStage.MATURE


def test_full_lifecycle():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    stages_seen = set()
    for _ in range(100):
        dc.tick()
        stages_seen.add(dc.stage)
    assert DevelopmentalStage.EMBRYONIC in stages_seen
    assert DevelopmentalStage.MATURE in stages_seen


def test_stage_never_regresses():
    t = Telomere(max_operations=100, allow_renewal=True)
    t.start()
    dc = DevelopmentController(telomere=t)
    for _ in range(50):
        dc.tick()
    stage_at_50 = dc.stage
    t.renew(50)  # Extend telomere — should NOT regress stage
    dc.tick()
    assert _STAGE_ORDER[dc.stage] >= _STAGE_ORDER[stage_at_50]


def test_on_stage_change_callback():
    transitions = []
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(
        telomere=t,
        on_stage_change=lambda old, new: transitions.append((old, new)),
    )
    for _ in range(100):
        dc.tick()
    assert len(transitions) >= 3  # EMBRYONIC→JUVENILE→ADOLESCENT→MATURE


# ---------------------------------------------------------------------------
# Learning plasticity
# ---------------------------------------------------------------------------


def test_learning_plasticity_decreases():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    initial = dc.learning_plasticity
    for _ in range(80):
        dc.tick()
    assert dc.learning_plasticity < initial


# ---------------------------------------------------------------------------
# Critical periods
# ---------------------------------------------------------------------------


def test_critical_period_open():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(
        telomere=t,
        critical_periods=(
            CriticalPeriod("fast_learning", DevelopmentalStage.EMBRYONIC, DevelopmentalStage.JUVENILE, "rapid adoption"),
        ),
    )
    assert dc.is_critical_period_open("fast_learning") is True


def test_critical_period_closes():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(
        telomere=t,
        critical_periods=(
            CriticalPeriod("fast_learning", DevelopmentalStage.EMBRYONIC, DevelopmentalStage.JUVENILE, "rapid adoption"),
        ),
    )
    for _ in range(15):  # Past juvenile threshold
        dc.tick()
    assert dc.is_critical_period_open("fast_learning") is False
    assert len(dc.closed_critical_periods()) >= 1


# ---------------------------------------------------------------------------
# can_acquire_stage + status
# ---------------------------------------------------------------------------


def test_can_acquire_stage():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    assert dc.can_acquire_stage(DevelopmentalStage.EMBRYONIC) is True
    assert dc.can_acquire_stage(DevelopmentalStage.MATURE) is False


def test_get_status():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    dc.tick()
    status = dc.get_status()
    assert status.stage == DevelopmentalStage.EMBRYONIC
    assert status.tick_count == 1


def test_get_statistics():
    t = Telomere(max_operations=100)
    t.start()
    dc = DevelopmentController(telomere=t)
    stats = dc.get_statistics()
    assert "stage" in stats
    assert "learning_plasticity" in stats
