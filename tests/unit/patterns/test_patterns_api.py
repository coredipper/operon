"""Tests for the pattern-first API layer."""

import pytest

from operon_ai import advise_topology, reviewer_gate, specialist_swarm
from operon_ai.core.epistemic import TopologyClass


def test_reviewer_gate_allows_custom_executor_and_reviewer():
    gate = reviewer_gate(
        executor=lambda prompt: f"EXECUTE::{prompt}",
        reviewer=lambda prompt, candidate: candidate.startswith("EXECUTE::"),
    )

    result = gate.run("Ship the patch")

    assert result.allowed is True
    assert result.output == "EXECUTE::Ship the patch"
    assert result.approval_token is not None
    assert gate.analysis.classification.topology_class == TopologyClass.CENTRALIZED


def test_reviewer_gate_blocks_when_reviewer_rejects():
    gate = reviewer_gate(
        executor=lambda prompt: f"EXECUTE::{prompt}",
        reviewer=lambda prompt, candidate: False,
    )

    result = gate.run("Drop production table")

    assert result.allowed is False
    assert result.status == "blocked"
    assert "reviewer" in result.reason.lower()


def test_reviewer_gate_cache_avoids_reexecution():
    calls = {"count": 0}

    def executor(prompt):
        calls["count"] += 1
        return f"EXECUTE::{prompt}"

    gate = reviewer_gate(
        executor=executor,
        reviewer=lambda prompt, candidate: True,
        enable_cache=True,
    )

    gate.run("Repeatable request")
    cached = gate.run("Repeatable request")

    assert calls["count"] == 1
    assert cached.raw.cached is True


def test_specialist_swarm_runs_and_analyzes():
    swarm = specialist_swarm(
        roles=["research", "risk"],
        workers={
            "research": lambda task, role: f"{role}: found two options",
            "risk": lambda task, role: f"{role}: no blockers found",
        },
        aggregator=lambda task, outputs: " || ".join(outputs.values()),
    )

    result = swarm.run("Assess vendor")

    assert result.outputs["research"] == "research: found two options"
    assert result.outputs["risk"] == "risk: no blockers found"
    assert result.aggregate == "research: found two options || risk: no blockers found"
    assert result.analysis.classification.topology_class == TopologyClass.CENTRALIZED


def test_specialist_swarm_single_argument_aggregator_receives_outputs():
    swarm = specialist_swarm(
        roles=["research", "risk"],
        workers={
            "research": lambda task, role: f"{role}: found two options",
            "risk": lambda task, role: f"{role}: no blockers found",
        },
        aggregator=lambda outputs: sorted(outputs.keys()),
    )

    result = swarm.run("Assess vendor")

    assert result.aggregate == ["research", "risk"]


def test_specialist_swarm_requires_unique_roles():
    with pytest.raises(ValueError, match="unique"):
        specialist_swarm(roles=["risk", "risk"])


def test_advise_topology_sequential_low_error_prefers_reviewer():
    advice = advise_topology(
        task_shape="sequential",
        tool_count=2,
        subtask_count=3,
        error_tolerance=0.02,
    )

    assert advice.recommended_pattern == "single_worker_with_reviewer"
    assert advice.suggested_api == "reviewer_gate(...)"


def test_advise_topology_parallel_prefers_swarm():
    advice = advise_topology(
        task_shape="parallel",
        tool_count=4,
        subtask_count=3,
        error_tolerance=0.1,
    )

    assert advice.recommended_pattern == "specialist_swarm"
    assert advice.suggested_api == "specialist_swarm(...)"


# ---------------------------------------------------------------------------
# Parallel stage groups
# ---------------------------------------------------------------------------


class TestParallelStageGroups:
    """Regression tests for parallel stage execution and merge logic."""

    def _make_organism(self, stages):
        from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism
        nucleus = Nucleus(provider=MockProvider())
        return skill_organism(
            stages=stages,
            fast_nucleus=nucleus,
            deep_nucleus=nucleus,
        )

    def test_flat_stages_are_sequential(self):
        from operon_ai import SkillStage
        org = self._make_organism([
            SkillStage(name="a", role="A", instructions="do A", mode="fixed"),
            SkillStage(name="b", role="B", instructions="do B", mode="fixed"),
        ])
        assert len(org.stage_groups) == 2
        assert all(len(g) == 1 for g in org.stage_groups)

    def test_grouped_stages_preserve_structure(self):
        from operon_ai import SkillStage
        org = self._make_organism([
            [
                SkillStage(name="a", role="A", instructions="do A", mode="fixed"),
                SkillStage(name="b", role="B", instructions="do B", mode="fixed"),
            ],
            SkillStage(name="c", role="C", instructions="do C", mode="fixed"),
        ])
        assert len(org.stage_groups) == 2
        assert len(org.stage_groups[0]) == 2  # parallel
        assert len(org.stage_groups[1]) == 1  # sequential

    def test_parallel_group_runs_all_stages(self):
        from operon_ai import SkillStage
        org = self._make_organism([[
            SkillStage(name="a", role="A", instructions="do A", mode="fixed"),
            SkillStage(name="b", role="B", instructions="do B", mode="fixed"),
            SkillStage(name="c", role="C", instructions="do C", mode="fixed"),
        ]])
        result = org.run("test")
        stage_names = [sr.stage_name for sr in result.stage_results]
        assert set(stage_names) == {"a", "b", "c"}
        assert len(result.stage_results) == 3

    def test_parallel_results_in_declared_order(self):
        from operon_ai import SkillStage
        org = self._make_organism([[
            SkillStage(name="x", role="X", instructions="do X", mode="fixed"),
            SkillStage(name="y", role="Y", instructions="do Y", mode="fixed"),
            SkillStage(name="z", role="Z", instructions="do Z", mode="fixed"),
        ]])
        result = org.run("test")
        stage_names = [sr.stage_name for sr in result.stage_results]
        assert stage_names == ["x", "y", "z"]

    def test_parallel_stage_outputs_merged(self):
        from operon_ai import SkillStage
        org = self._make_organism([[
            SkillStage(name="a", role="A",
                       handler=lambda t, s, o, st: "output_a", mode="fixed"),
            SkillStage(name="b", role="B",
                       handler=lambda t, s, o, st: "output_b", mode="fixed"),
        ]])
        result = org.run("test")
        assert result.shared_state.get("a") == "output_a"
        assert result.shared_state.get("b") == "output_b"

    def test_internal_list_key_merged_from_parallel_stages(self):
        """Regression: new _-prefixed list keys created in parallel stages
        must be merged (concatenated), not overwritten by last writer.

        Tests the merge function directly since handlers get a copy of
        state and can't mutate shared_state.
        """
        from operon_ai.patterns.organism import _merge_parallel_results

        snap: dict = {}
        per_stage = {
            "a": ({"_test_signals": ["signal_a"]}, {"a": "out"}, [], "continue"),
            "b": ({"_test_signals": ["signal_b"]}, {"b": "out"}, [], "continue"),
        }
        state: dict = {}
        stage_outputs: dict = {}
        stage_results: list = []

        _merge_parallel_results(state, stage_outputs, stage_results, snap, per_stage)

        signals = state.get("_test_signals", [])
        # Exact order: declared stage order (a before b)
        assert signals == ["signal_a", "signal_b"]

    def test_state_conflict_raises(self):
        """Two parallel stages writing different values to the same
        non-internal key should raise StateConflictError."""
        from operon_ai import SkillStage
        from operon_ai.patterns.types import StateConflictError

        # Note: handlers get a COPY of state, so direct state mutation
        # doesn't propagate. But component hooks DO write to shared_state.
        # For this test, we test the merge function directly.
        from operon_ai.patterns.organism import _merge_parallel_results

        snap = {}
        per_stage = {
            "a": ({"user_key": "value_1"}, {"a": "out"}, [], "continue"),
            "b": ({"user_key": "value_2"}, {"b": "out"}, [], "continue"),
        }
        state: dict = {}
        stage_outputs: dict = {}
        stage_results: list = []

        with pytest.raises(StateConflictError, match="user_key"):
            _merge_parallel_results(state, stage_outputs, stage_results, snap, per_stage)

    def test_internal_list_key_e2e_with_component(self):
        """E2E: a real component appends to an internal list key during
        parallel execution via org.run(), not just the merge function."""
        from dataclasses import dataclass, field
        from operon_ai import SkillStage

        @dataclass
        class SignalCollector:
            """Minimal component that appends to _test_signals."""
            signals: list = field(default_factory=list)

            def on_run_start(self, task, shared_state):
                self.signals.clear()

            def on_stage_start(self, stage, shared_state, stage_outputs):
                pass

            def on_stage_result(self, stage, result, shared_state, stage_outputs):
                shared_state.setdefault("_test_signals", []).append(
                    f"signal_{stage.name}"
                )

            def on_run_complete(self, result, shared_state):
                pass

        collector = SignalCollector()
        from operon_ai import MockProvider, Nucleus, skill_organism
        nucleus = Nucleus(provider=MockProvider())
        org = skill_organism(
            stages=[[
                SkillStage(name="a", role="A", instructions="do A", mode="fixed"),
                SkillStage(name="b", role="B", instructions="do B", mode="fixed"),
            ]],
            fast_nucleus=nucleus,
            deep_nucleus=nucleus,
            components=[collector],
        )
        result = org.run("test")
        signals = result.shared_state.get("_test_signals", [])
        # Exact declared order
        assert signals == ["signal_a", "signal_b"]

    def test_parallel_speedup(self):
        """Parallel group should complete faster than sequential."""
        import time
        from operon_ai import SkillStage

        sleep_ms = 100

        def slow_handler(task, state, outputs, stage):
            time.sleep(sleep_ms / 1000)
            return "done"

        # Parallel: 3 stages × 100ms = ~100ms wall-clock
        org_par = self._make_organism([[
            SkillStage(name="a", role="A", handler=slow_handler, mode="fixed"),
            SkillStage(name="b", role="B", handler=slow_handler, mode="fixed"),
            SkillStage(name="c", role="C", handler=slow_handler, mode="fixed"),
        ]])

        t0 = time.monotonic()
        org_par.run("test")
        parallel_ms = (time.monotonic() - t0) * 1000

        # Sequential: 3 stages × 100ms = ~300ms wall-clock
        org_seq = self._make_organism([
            SkillStage(name="a", role="A", handler=slow_handler, mode="fixed"),
            SkillStage(name="b", role="B", handler=slow_handler, mode="fixed"),
            SkillStage(name="c", role="C", handler=slow_handler, mode="fixed"),
        ])

        t0 = time.monotonic()
        org_seq.run("test")
        sequential_ms = (time.monotonic() - t0) * 1000

        # Parallel should be at least 1.5x faster than sequential
        speedup = sequential_ms / max(parallel_ms, 1)
        assert speedup > 1.5, (
            f"Expected speedup > 1.5x, got {speedup:.1f}x "
            f"(parallel={parallel_ms:.0f}ms, sequential={sequential_ms:.0f}ms)"
        )

    def test_backward_compatible_run(self):
        """Flat-list organism produces identical results to pre-parallel behavior."""
        from operon_ai import SkillStage
        org = self._make_organism([
            SkillStage(name="s1", role="R1",
                       handler=lambda t, s, o, st: "r1", mode="fixed"),
            SkillStage(name="s2", role="R2",
                       handler=lambda t, s, o, st: "r2", mode="fixed"),
        ])
        result = org.run("test")
        assert [sr.stage_name for sr in result.stage_results] == ["s1", "s2"]
        assert result.shared_state.get("s1") == "r1"
        assert result.shared_state.get("s2") == "r2"
        assert result.shared_state.get("last_stage") == "s2"
