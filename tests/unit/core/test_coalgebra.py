"""Tests for coalgebraic state machines (Paper §4.2)."""

import pytest

from operon_ai.core.coalgebra import (
    BisimulationResult,
    Coalgebra,
    FunctionalCoalgebra,
    ParallelCoalgebra,
    SequentialCoalgebra,
    StateMachine,
    TransitionRecord,
    check_bisimulation,
    counter_coalgebra,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _doubler() -> FunctionalCoalgebra[int, int, int]:
    """State tracks cumulative product; readout = state; update = state * input."""
    return FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s * i,
    )


# ── TestCoalgebraProtocol ────────────────────────────────────────────────


class TestCoalgebraProtocol:
    def test_functional_coalgebra_satisfies_protocol(self):
        c = counter_coalgebra()
        assert isinstance(c, Coalgebra)

    def test_readout(self):
        c = counter_coalgebra()
        assert c.readout(10) == 10

    def test_update(self):
        c = counter_coalgebra()
        assert c.update(10, 5) == 15


# ── TestFunctionalCoalgebra ─────────────────────────────────────────────


class TestFunctionalCoalgebra:
    def test_frozen(self):
        c = counter_coalgebra()
        with pytest.raises(AttributeError):
            c.readout_fn = lambda s: s  # type: ignore[misc]

    def test_custom_functions(self):
        c = FunctionalCoalgebra(
            readout_fn=lambda s: s * 2,
            update_fn=lambda s, i: s - i,
        )
        assert c.readout(5) == 10
        assert c.update(5, 3) == 2


# ── TestStateMachine ────────────────────────────────────────────────────


class TestStateMachine:
    def test_step_returns_output(self):
        sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        out = sm.step(5)
        assert out == 0  # readout of state *before* update

    def test_step_updates_state(self):
        sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        sm.step(5)
        assert sm.state == 5

    def test_run_sequence(self):
        sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        outputs = sm.run([1, 2, 3])
        assert outputs == [0, 1, 3]  # readout before each update
        assert sm.state == 6

    def test_trace_recording(self):
        sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        sm.step(5)
        assert len(sm.trace) == 1
        rec = sm.trace[0]
        assert isinstance(rec, TransitionRecord)
        assert rec.state_before == 0
        assert rec.input == 5
        assert rec.output == 0
        assert rec.state_after == 5
        assert rec.step == 1

    def test_trace_disabled(self):
        sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        sm.step(5, record=False)
        assert len(sm.trace) == 0
        assert sm.state == 5

    def test_reset_trace(self):
        sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        sm.run([1, 2, 3])
        assert len(sm.trace) == 3
        sm.reset_trace()
        assert len(sm.trace) == 0


# ── TestParallelCoalgebra ───────────────────────────────────────────────


class TestParallelCoalgebra:
    def test_parallel_readout(self):
        p = ParallelCoalgebra(first=counter_coalgebra(), second=_doubler())
        assert p.readout((10, 3)) == (10, 3)

    def test_parallel_update(self):
        p = ParallelCoalgebra(first=counter_coalgebra(), second=_doubler())
        new_state = p.update((10, 3), 2)
        assert new_state == (12, 6)  # 10+2, 3*2


# ── TestSequentialCoalgebra ─────────────────────────────────────────────


class TestSequentialCoalgebra:
    def test_sequential_composition(self):
        # first: counter (readout=state, update=state+inp)
        # second: doubler (readout=state, update=state*inp)
        # intermediate = first.readout(s1) fed to second.update
        seq = SequentialCoalgebra(first=counter_coalgebra(), second=_doubler())
        # Initial state (0, 1): first.readout(0)=0, used as second input
        # first: 0+5=5, second: 1*0=0
        new_state = seq.update((0, 1), 5)
        assert new_state == (5, 0)

    def test_sequential_readout(self):
        seq = SequentialCoalgebra(first=counter_coalgebra(), second=_doubler())
        assert seq.readout((10, 3)) == (10, 3)


# ── TestBisimulation ───────────────────────────────────────────────────


class TestBisimulation:
    def test_identical_machines_equivalent(self):
        a = StateMachine(state=0, coalgebra=counter_coalgebra())
        b = StateMachine(state=0, coalgebra=counter_coalgebra())
        result = check_bisimulation(a, b, [1, 2, 3, 4, 5])
        assert result.equivalent is True
        assert result.witness is None
        assert result.states_explored == 5

    def test_different_initial_state_not_equivalent(self):
        a = StateMachine(state=0, coalgebra=counter_coalgebra())
        b = StateMachine(state=100, coalgebra=counter_coalgebra())
        result = check_bisimulation(a, b, [1, 2, 3])
        assert result.equivalent is False
        assert result.witness is not None

    def test_witness_on_mismatch(self):
        a = StateMachine(state=0, coalgebra=counter_coalgebra())
        b = StateMachine(state=0, coalgebra=_doubler())
        result = check_bisimulation(a, b, [1, 2, 3])
        assert result.equivalent is False
        inp, out_a, out_b = result.witness
        # First step: both readout(0)=0, so they match
        # After step 1: a.state=1, b.state=0
        # Second step: a.readout(1)=1, b.readout(0)=0 → diverge
        assert result.states_explored == 2

    def test_empty_inputs(self):
        a = StateMachine(state=0, coalgebra=counter_coalgebra())
        b = StateMachine(state=999, coalgebra=_doubler())
        result = check_bisimulation(a, b, [])
        assert result.equivalent is True
        assert result.states_explored == 0
