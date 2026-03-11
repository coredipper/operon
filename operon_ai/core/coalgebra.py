"""
Coalgebraic State Machines: Composable Observation & Evolution
==============================================================

Paper §3.5: Agents as state machines with coalgebraic structure.

Biological Analogy:
A cell's internal state is never accessed directly — it is *observed*
through surface markers (readout) and *evolved* through signal
transduction (update).  This observation-based view is precisely a
coalgebra: state → observable output, (state, input) → next state.

Existing Operon components (HistoneStore, ATP_Store, CellCycleController)
already follow this pattern implicitly.  This module makes it explicit,
enabling:

1. Composition  — parallel and sequential state machines
2. Bisimulation — formal equivalence checking between machines
3. Tracing      — full transition history for debugging & audit

Key types:
- Coalgebra       — Protocol: readout + update
- FunctionalCoalgebra — Concrete coalgebra from two plain functions
- ParallelCoalgebra   — Product of two coalgebras (shared input)
- SequentialCoalgebra — Pipeline: output of first feeds input of second
- StateMachine    — Wrapper that manages current state and trace
- TransitionRecord — Single step in a trace
- BisimulationResult — Outcome of a bisimulation check

References:
- Article Section 3.5: Epigenetics and State - The Coalgebra
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

S = TypeVar("S")
I = TypeVar("I")
O = TypeVar("O")

S1 = TypeVar("S1")
S2 = TypeVar("S2")
I1 = TypeVar("I1")
I2 = TypeVar("I2")
O1 = TypeVar("O1")
O2 = TypeVar("O2")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Coalgebra(Protocol[S, I, O]):
    """
    Coalgebraic interface: observe state (readout) and evolve it (update).

    A Coalgebra[S, I, O] defines a Mealy machine where:
    - S is the state space
    - I is the input alphabet
    - O is the output alphabet
    - readout: S → O   (observation)
    - update:  S × I → S  (transition)
    """

    def readout(self, state: S) -> O: ...
    def update(self, state: S, inp: I) -> S: ...


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass
class TransitionRecord(Generic[S, I, O]):
    """A single recorded transition of a state machine."""

    state_before: S
    input: I
    output: O
    state_after: S
    step: int


@dataclass
class BisimulationResult:
    """Outcome of a bisimulation equivalence check."""

    equivalent: bool
    witness: tuple[Any, ...] | None  # (input, output_a, output_b) on mismatch
    states_explored: int
    message: str


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalCoalgebra(Generic[S, I, O]):
    """
    A coalgebra defined by two plain functions.

    This is the simplest concrete implementation: supply a readout
    function and an update function and you have a coalgebra.

    Example:
        >>> counter = FunctionalCoalgebra(
        ...     readout_fn=lambda s: s,
        ...     update_fn=lambda s, i: s + i,
        ... )
        >>> counter.readout(0)
        0
        >>> counter.update(0, 5)
        5
    """

    readout_fn: Callable[[S], O]
    update_fn: Callable[[S, I], S]

    def readout(self, state: S) -> O:
        return self.readout_fn(state)

    def update(self, state: S, inp: I) -> S:
        return self.update_fn(state, inp)


@dataclass(frozen=True)
class ParallelCoalgebra(Generic[S1, S2, I, O1, O2]):
    """
    Product of two coalgebras running in parallel on the same input.

    State is (S1, S2), output is (O1, O2).  Both coalgebras receive
    the same input and evolve independently.

    Biological Analogy:
    Two signaling pathways activated by the same ligand — e.g.,
    MAPK and PI3K both triggered by EGF binding.
    """

    first: Coalgebra[S1, I, O1]
    second: Coalgebra[S2, I, O2]

    def readout(self, state: tuple[S1, S2]) -> tuple[O1, O2]:
        s1, s2 = state
        return (self.first.readout(s1), self.second.readout(s2))

    def update(self, state: tuple[S1, S2], inp: I) -> tuple[S1, S2]:
        s1, s2 = state
        return (self.first.update(s1, inp), self.second.update(s2, inp))


@dataclass(frozen=True)
class SequentialCoalgebra(Generic[S1, S2, I, O1, O2]):
    """
    Sequential composition: output of first feeds as input to second.

    State is (S1, S2).  Input goes to the first coalgebra; its
    current readout becomes the input to the second coalgebra's
    update.  The composite readout returns both component readouts,
    preserving observability of the joint state.

    Biological Analogy:
    Signal transduction cascade — receptor activation (first) produces
    a second messenger that activates an effector (second).
    """

    first: Coalgebra[S1, I, O1]
    second: Coalgebra[S2, O1, O2]

    def readout(self, state: tuple[S1, S2]) -> tuple[O1, O2]:
        s1, s2 = state
        return (self.first.readout(s1), self.second.readout(s2))

    def update(self, state: tuple[S1, S2], inp: I) -> tuple[S1, S2]:
        s1, s2 = state
        intermediate = self.first.readout(s1)
        new_s1 = self.first.update(s1, inp)
        new_s2 = self.second.update(s2, intermediate)
        return (new_s1, new_s2)


# ---------------------------------------------------------------------------
# State machine wrapper
# ---------------------------------------------------------------------------


@dataclass
class StateMachine(Generic[S, I, O]):
    """
    A running state machine with current state and optional trace.

    Wraps a Coalgebra with mutable state, providing a convenient
    imperative interface for stepping and running sequences.

    Example:
        >>> from operon_ai.core.coalgebra import counter_coalgebra, StateMachine
        >>> sm = StateMachine(state=0, coalgebra=counter_coalgebra())
        >>> sm.step(3)
        3
        >>> sm.step(2)
        5
        >>> sm.state
        5
    """

    state: S
    coalgebra: Coalgebra[S, I, O]
    trace: list[TransitionRecord[S, I, O]] = field(default_factory=list)
    _step_count: int = field(default=0, repr=False)

    def readout(self) -> O:
        """Observe the current state without modifying it."""
        return self.coalgebra.readout(self.state)

    def step(self, inp: I, *, record: bool = True) -> O:
        """
        Apply one input, update state, return output.

        If record=True, the transition is appended to the trace.
        """
        output = self.coalgebra.readout(self.state)
        state_before = self.state
        self.state = self.coalgebra.update(self.state, inp)
        self._step_count += 1

        if record:
            self.trace.append(
                TransitionRecord(
                    state_before=state_before,
                    input=inp,
                    output=output,
                    state_after=self.state,
                    step=self._step_count,
                )
            )
        return output

    def run(self, inputs: list[I], *, record: bool = True) -> list[O]:
        """Run a sequence of inputs, returning the list of outputs."""
        return [self.step(inp, record=record) for inp in inputs]

    def reset_trace(self) -> None:
        """Clear the transition trace."""
        self.trace.clear()


# ---------------------------------------------------------------------------
# Bisimulation
# ---------------------------------------------------------------------------


def check_bisimulation(
    machine_a: StateMachine[Any, Any, Any],
    machine_b: StateMachine[Any, Any, Any],
    inputs: list[Any],
) -> BisimulationResult:
    """
    Check observational equivalence of two machines over a sequence of inputs.

    Two machines are bisimilar if, for every input in the sequence,
    they produce the same output.  If they diverge, the first
    diverging input is returned as a *witness*.

    This is a *bounded* bisimulation check — it only tests the
    supplied input sequence, not all possible inputs.

    Args:
        machine_a: First state machine
        machine_b: Second state machine
        inputs:    Sequence of inputs to test

    Returns:
        BisimulationResult with equivalence verdict and optional witness.
    """
    if not inputs:
        return BisimulationResult(
            equivalent=True,
            witness=None,
            states_explored=0,
            message="No inputs to test — vacuously equivalent.",
        )

    for i, inp in enumerate(inputs):
        out_a = machine_a.step(inp, record=False)
        out_b = machine_b.step(inp, record=False)
        if out_a != out_b:
            return BisimulationResult(
                equivalent=False,
                witness=(inp, out_a, out_b),
                states_explored=i + 1,
                message=(
                    f"Diverged at step {i + 1}: "
                    f"input={inp!r}, output_a={out_a!r}, output_b={out_b!r}"
                ),
            )

    return BisimulationResult(
        equivalent=True,
        witness=None,
        states_explored=len(inputs),
        message=f"Equivalent over {len(inputs)} inputs.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def counter_coalgebra() -> FunctionalCoalgebra[int, int, int]:
    """
    A simple counter coalgebra for demos and testing.

    State = running total (int).  Input = delta (int).
    readout = current total.  update = state + delta.
    """
    return FunctionalCoalgebra(
        readout_fn=lambda s: s,
        update_fn=lambda s, i: s + i,
    )
