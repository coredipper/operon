"""Tests for the C6 convergence evaluation harness.

Covers task definitions, configurations, metrics, mock evaluator,
harness orchestration, structural variation, credit assignment, and
report generation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eval.convergence.configurations import ConfigurationSpec, get_configurations
from eval.convergence.credit_assignment import (
    StageCredit,
    aggregate_credit,
    assign_credit,
)
from eval.convergence.harness import ConvergenceHarness, HarnessConfig
from eval.convergence.metrics import (
    AggregateMetrics,
    RunMetrics,
    collect_metrics,
    compare_configs,
)
from eval.convergence.mock_evaluator import MockEvaluator
from eval.convergence.report import generate_convergence_report, ranking_table
from eval.convergence.structural_variation import topology_distance, variation_summary
from eval.convergence.tasks import (
    TaskDefinition,
    get_benchmark_tasks,
    task_to_fingerprint,
)


# ===================================================================
# Task tests
# ===================================================================


class TestTasks:
    def test_benchmark_tasks_count(self):
        tasks = get_benchmark_tasks()
        assert len(tasks) == 20

    def test_task_ids_unique(self):
        tasks = get_benchmark_tasks()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_difficulty_distribution(self):
        tasks = get_benchmark_tasks()
        easy = [t for t in tasks if t.difficulty == "easy"]
        medium = [t for t in tasks if t.difficulty == "medium"]
        hard = [t for t in tasks if t.difficulty == "hard"]
        assert len(easy) == 5
        assert len(medium) == 8
        assert len(hard) == 7

    def test_shape_distribution(self):
        tasks = get_benchmark_tasks()
        seq = [t for t in tasks if t.task_shape == "sequential"]
        mixed = [t for t in tasks if t.task_shape == "mixed"]
        par = [t for t in tasks if t.task_shape == "parallel"]
        assert len(seq) == 5
        assert len(mixed) == 8
        assert len(par) == 7

    def test_task_to_fingerprint(self):
        task = get_benchmark_tasks()[0]
        fp = task_to_fingerprint(task)
        assert fp.task_shape == task.task_shape
        assert fp.tool_count == task.tool_count
        assert fp.subtask_count == task.subtask_count
        assert fp.required_roles == task.required_roles

    def test_all_tasks_have_roles(self):
        for task in get_benchmark_tasks():
            assert len(task.required_roles) > 0
            assert task.subtask_count >= 2


# ===================================================================
# Configuration tests
# ===================================================================


class TestConfigurations:
    def test_configurations_count(self):
        configs = get_configurations()
        assert len(configs) == 7

    def test_config_ids_unique(self):
        configs = get_configurations()
        ids = [c.config_id for c in configs]
        assert len(ids) == len(set(ids))

    def test_guided_configs_exist(self):
        configs = get_configurations()
        guided = [c for c in configs if c.structural_guidance]
        assert len(guided) >= 2  # at least swarms_operon, scion_operon, operon_adaptive

    def test_baseline_configs_exist(self):
        configs = get_configurations()
        baselines = [c for c in configs if not c.structural_guidance]
        assert len(baselines) >= 3

    def test_framework_coverage(self):
        configs = get_configurations()
        frameworks = {c.framework for c in configs}
        assert "swarms" in frameworks
        assert "deerflow" in frameworks
        assert "ralph" in frameworks
        assert "scion" in frameworks
        assert "operon" in frameworks


# ===================================================================
# Metrics tests
# ===================================================================


class TestMetrics:
    def _make_run(self, task_id: str, config_id: str, success: bool) -> RunMetrics:
        return RunMetrics(
            task_id=task_id,
            config_id=config_id,
            success=success,
            token_cost=1000,
            latency_ms=500.0,
            intervention_count=1,
            convergence_rate=0.8,
            structural_variation=0.1,
            risk_score=0.2,
            stage_count=3,
        )

    def test_collect_metrics_grouping(self):
        runs = [
            self._make_run("t1", "c1", True),
            self._make_run("t2", "c1", False),
            self._make_run("t1", "c2", True),
        ]
        agg = collect_metrics(runs)
        assert "c1" in agg
        assert "c2" in agg
        assert agg["c1"].n_tasks == 2
        assert agg["c2"].n_tasks == 1
        assert agg["c1"].success_rate == 0.5
        assert agg["c2"].success_rate == 1.0

    def test_compare_configs_ordering(self):
        runs = [
            self._make_run("t1", "good", True),
            self._make_run("t2", "good", True),
            self._make_run("t1", "bad", False),
            self._make_run("t2", "bad", False),
        ]
        agg = collect_metrics(runs)
        comparison = compare_configs(agg)
        assert comparison[0]["config_id"] == "good"
        assert comparison[1]["config_id"] == "bad"

    def test_empty_runs(self):
        agg = collect_metrics([])
        assert agg == {}


# ===================================================================
# Structural variation tests
# ===================================================================


class TestStructuralVariation:
    def test_same_shape_zero_distance(self):
        assert topology_distance("sequential", "sequential") == 0.0
        assert topology_distance("parallel", "parallel") == 0.0

    def test_adjacent_shapes(self):
        assert topology_distance("sequential", "mixed") == 0.5
        assert topology_distance("mixed", "parallel") == 0.5

    def test_max_distance(self):
        assert topology_distance("sequential", "parallel") == 1.0

    def test_variation_summary(self):
        runs = [
            RunMetrics("t1", "c1", True, 0, 0, 0, 0, 0.5, 0, 0),
            RunMetrics("t2", "c1", True, 0, 0, 0, 0, 0.3, 0, 0),
            RunMetrics("t1", "c2", True, 0, 0, 0, 0, 0.0, 0, 0),
        ]
        summary = variation_summary(runs)
        assert abs(summary["c1"] - 0.4) < 1e-6
        assert summary["c2"] == 0.0


# ===================================================================
# Credit assignment tests
# ===================================================================


class TestCreditAssignment:
    def test_assign_credit_success(self):
        import random
        rng = random.Random(42)
        credits = assign_credit(("writer", "reviewer"), True, 0.2, rng)
        assert len(credits) == 2
        assert all(c.contribution > 0 for c in credits)

    def test_assign_credit_failure(self):
        import random
        rng = random.Random(42)
        credits = assign_credit(("writer", "reviewer"), False, 0.8, rng)
        assert len(credits) == 2
        assert all(c.contribution < 0 for c in credits)

    def test_assign_credit_empty_roles(self):
        import random
        rng = random.Random(42)
        credits = assign_credit((), True, 0.2, rng)
        assert credits == []

    def test_aggregate_credit(self):
        credits1 = [
            StageCredit("s0", "writer", 0.6, False),
            StageCredit("s1", "reviewer", 0.4, False),
        ]
        credits2 = [
            StageCredit("s0", "writer", 0.3, True),
            StageCredit("s1", "reviewer", 0.7, False),
        ]
        agg = aggregate_credit([credits1, credits2])
        assert abs(agg["writer"] - 0.9) < 1e-3
        assert abs(agg["reviewer"] - 1.1) < 1e-3


# ===================================================================
# Mock evaluator tests
# ===================================================================


class TestMockEvaluator:
    def test_deterministic(self):
        """Same seed produces identical results."""
        task = get_benchmark_tasks()[0]
        config = get_configurations()[0]

        ev1 = MockEvaluator(seed=42)
        ev2 = MockEvaluator(seed=42)

        r1 = ev1.evaluate(task, config)
        r2 = ev2.evaluate(task, config)

        assert r1.success == r2.success
        assert r1.risk_score == r2.risk_score
        assert r1.token_cost == r2.token_cost
        assert r1.latency_ms == r2.latency_ms

    def test_guided_lower_risk(self):
        """Guided configs should produce lower risk scores than baselines."""
        task = get_benchmark_tasks()[5]  # medium difficulty
        configs = get_configurations()

        swarms_baseline = next(c for c in configs if c.config_id == "swarms_baseline")
        swarms_guided = next(c for c in configs if c.config_id == "swarms_operon")

        ev = MockEvaluator(seed=1337)
        baseline_result = ev.evaluate(task, swarms_baseline)
        guided_result = ev.evaluate(task, swarms_guided)

        # Guided should have <= baseline risk (30% reduction applied).
        assert guided_result.risk_score <= baseline_result.risk_score

    def test_all_frameworks_compile(self):
        """Every configuration should produce a valid RunMetrics."""
        task = get_benchmark_tasks()[0]
        ev = MockEvaluator(seed=99)
        for config in get_configurations():
            result = ev.evaluate(task, config)
            assert isinstance(result, RunMetrics)
            assert 0.0 <= result.risk_score <= 1.0
            assert result.token_cost > 0
            assert result.latency_ms > 0

    def test_risk_score_bounded(self):
        """Risk scores should always be in [0, 1]."""
        ev = MockEvaluator(seed=1337)
        tasks = get_benchmark_tasks()[:3]
        for task in tasks:
            for config in get_configurations():
                r = ev.evaluate(task, config)
                assert 0.0 <= r.risk_score <= 1.0


# ===================================================================
# Harness tests
# ===================================================================


class TestHarness:
    def test_full_run(self):
        """Harness runs to completion with default config."""
        config = HarnessConfig(seed=42, tasks=["easy_seq_01"], configs=["swarms_baseline"])
        harness = ConvergenceHarness(config)
        results = harness.run()

        assert results["seed"] == 42
        assert results["n_tasks"] == 1
        assert results["n_configs"] == 1
        assert len(results["runs"]) == 1
        assert "swarms_baseline" in results["aggregates"]

    def test_task_filter(self):
        """Task filter selects only specified tasks."""
        config = HarnessConfig(tasks=["easy_seq_01", "easy_seq_02"])
        harness = ConvergenceHarness(config)
        assert len(harness.tasks) == 2
        assert all(t.task_id in ("easy_seq_01", "easy_seq_02") for t in harness.tasks)

    def test_config_filter(self):
        """Config filter selects only specified configurations."""
        config = HarnessConfig(configs=["swarms_baseline", "operon_adaptive"])
        harness = ConvergenceHarness(config)
        assert len(harness.configurations) == 2

    def test_multi_task_multi_config(self):
        """Harness produces correct number of runs for N tasks x M configs."""
        config = HarnessConfig(
            seed=99,
            tasks=["easy_seq_01", "easy_seq_02", "med_mix_01"],
            configs=["swarms_baseline", "swarms_operon", "operon_adaptive"],
        )
        harness = ConvergenceHarness(config)
        results = harness.run()

        assert len(results["runs"]) == 3 * 3  # 3 tasks x 3 configs
        assert results["n_tasks"] == 3
        assert results["n_configs"] == 3
        assert len(results["comparison"]) == 3

    def test_harness_comparison_sorted(self):
        """Comparison table is sorted by success rate descending."""
        config = HarnessConfig(
            seed=1337,
            tasks=["easy_seq_01", "med_mix_01", "hard_par_01"],
            configs=["swarms_baseline", "swarms_operon", "operon_adaptive"],
        )
        harness = ConvergenceHarness(config)
        results = harness.run()
        comparison = results["comparison"]

        # Verify sorted by success_rate desc.
        for i in range(len(comparison) - 1):
            assert comparison[i]["success_rate"] >= comparison[i + 1]["success_rate"] or (
                comparison[i]["success_rate"] == comparison[i + 1]["success_rate"]
                and comparison[i]["mean_risk_score"] <= comparison[i + 1]["mean_risk_score"]
            )


# ===================================================================
# Report tests
# ===================================================================


class TestReport:
    def test_ranking_table_output(self):
        agg = {
            "c1": AggregateMetrics("c1", 0.9, 1000, 500, 1.0, 0.1, 10),
            "c2": AggregateMetrics("c2", 0.5, 2000, 800, 3.0, 0.5, 10),
        }
        table = ranking_table(agg)
        assert "c1" in table
        assert "c2" in table
        assert "Rank" in table
        # c1 should rank higher (better success rate).
        c1_pos = table.index("c1")
        c2_pos = table.index("c2")
        assert c1_pos < c2_pos

    def test_generate_report_json(self):
        config = HarnessConfig(
            seed=42,
            tasks=["easy_seq_01"],
            configs=["swarms_baseline"],
        )
        harness = ConvergenceHarness(config)
        results = harness.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = str(Path(tmpdir) / "results.json")
            generate_convergence_report(results, out_json=json_path)

            loaded = json.loads(Path(json_path).read_text())
            assert loaded["seed"] == 42
            assert len(loaded["runs"]) == 1

    def test_generate_report_markdown(self):
        config = HarnessConfig(
            seed=42,
            tasks=["easy_seq_01"],
            configs=["swarms_baseline"],
        )
        harness = ConvergenceHarness(config)
        results = harness.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = str(Path(tmpdir) / "report.md")
            generate_convergence_report(results, out_markdown=md_path)

            content = Path(md_path).read_text()
            assert "C6 Convergence Evaluation Report" in content
            assert "swarms_baseline" in content
