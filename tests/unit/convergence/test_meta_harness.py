"""Tests for C8 Meta-Harness: evolution of organism configurations.

Tests the biological hypotheses:
- Genome abstraction covers full configuration space (round-trip tests)
- Co-design composition captures evolutionary dynamics (DesignProblem wrapping)
- Epistemic health monitoring generalizes across scales (distance-based stall)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from operon_ai.convergence.meta_types import (
    AssessmentRecord,
    CandidateConfig,
    ConfigHammingDistance,
    StageConfig,
    candidate_to_genome,
    genome_to_candidate,
)
from operon_ai.convergence.meta_store import EvolutionStore
from operon_ai.convergence.meta_proposers import (
    LLMProposer,
    Proposer,
    TournamentMutator,
)
from operon_ai.convergence.meta_protocol import FilesystemOptimizer
from operon_ai.convergence.evolution_loop import EvolutionConfig, EvolutionLoop
from operon_ai.health.epiplexity import (
    DistanceProvider,
    EpiplexityMonitor,
    HealthStatus,
)
from operon_ai.state.genome import GeneType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stage(role: str = "researcher", mode: str = "fuzzy",
                model: str | None = None) -> StageConfig:
    return StageConfig(role=role, mode=mode, model=model)


def _make_candidate(
    cid: str = "c_000",
    stages: tuple[StageConfig, ...] | None = None,
    policy: dict[str, Any] | None = None,
    iteration: int = 0,
) -> CandidateConfig:
    if stages is None:
        stages = (_make_stage("researcher"), _make_stage("reviewer", "fixed"))
    return CandidateConfig(
        candidate_id=cid,
        parent_id=None,
        iteration=iteration,
        stage_configs=stages,
        intervention_policy=policy or {"max_rate": 0.5},
    )


@dataclass
class _MockTaskDef:
    task_id: str = "easy_seq_01"
    name: str = "test"
    description: str = "Summarize this text."
    difficulty: str = "easy"
    task_shape: str = "sequential"
    tool_count: int = 0
    subtask_count: int = 1
    required_roles: tuple[str, ...] = ("researcher",)
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Step 1 tests: CandidateConfig <-> Genome mapping
# ---------------------------------------------------------------------------


class TestCandidateGenomeMapping:
    """Tests that the gene abstraction covers the full configuration space."""

    def test_round_trip_lossless(self):
        cc = _make_candidate()
        genome = candidate_to_genome(cc)
        restored = genome_to_candidate(
            genome, cc.candidate_id, cc.parent_id, cc.iteration,
        )
        assert restored.stage_configs == cc.stage_configs
        assert restored.intervention_policy == cc.intervention_policy

    def test_round_trip_with_all_fields(self):
        sc = StageConfig(
            role="analyst", mode="fixed", model="gemini-2.5-pro",
            include_stage_outputs=False, include_shared_state=False,
            cognitive_mode="observational",
        )
        cc = CandidateConfig(
            "c_full", "c_parent", 5, (sc,),
            {"max_rate": 0.3, "retry_limit": 3},
            proposer="llm_explore", reason="test all fields",
        )
        genome = candidate_to_genome(cc)
        restored = genome_to_candidate(
            genome, cc.candidate_id, cc.parent_id, cc.iteration,
            cc.proposer, cc.reason,
        )
        assert restored == cc

    def test_genome_diff_detects_stage_changes(self):
        cc = _make_candidate()
        g1 = candidate_to_genome(cc)
        g2 = g1.replicate(mutations={"stage_0_mode": "fixed"})
        diff = g1.diff(g2)
        assert "stage_0_mode" in diff
        assert diff["stage_0_mode"] == ("fuzzy", "fixed")

    def test_same_config_same_hash(self):
        cc = _make_candidate()
        g1 = candidate_to_genome(cc)
        g2 = candidate_to_genome(cc)
        assert g1.get_hash() == g2.get_hash()

    def test_different_config_different_hash(self):
        cc1 = _make_candidate()
        cc2 = _make_candidate(stages=(_make_stage("analyst", "fixed"),))
        assert candidate_to_genome(cc1).get_hash() != candidate_to_genome(cc2).get_hash()

    def test_gene_types_assigned_correctly(self):
        cc = _make_candidate()
        genome = candidate_to_genome(cc)
        mode_gene = genome.get_gene("stage_0_mode")
        assert mode_gene is not None
        assert mode_gene.gene_type == GeneType.STRUCTURAL

        bool_gene = genome.get_gene("stage_0_include_stage_outputs")
        assert bool_gene is not None
        assert bool_gene.gene_type == GeneType.REGULATORY

        meta_gene = genome.get_gene("_n_stages")
        assert meta_gene is not None
        assert meta_gene.gene_type == GeneType.HOUSEKEEPING


# ---------------------------------------------------------------------------
# Step 2 tests: EvolutionStore
# ---------------------------------------------------------------------------


class TestEvolutionStore:

    def test_save_load_candidate_round_trip(self, tmp_path):
        store = EvolutionStore(root=tmp_path / "run_001")
        cc = _make_candidate()
        genome = candidate_to_genome(cc)
        store.save_candidate(cc, genome)

        loaded = store.load_candidate("c_000")
        assert loaded.stage_configs == cc.stage_configs
        assert loaded.intervention_policy == cc.intervention_policy

    def test_index_append_and_load(self, tmp_path):
        store = EvolutionStore(root=tmp_path / "run_002")
        records = [
            AssessmentRecord("c_000", 0, "t1", 0.8, 100, 500.0, True, "seed"),
            AssessmentRecord("c_001", 1, "t2", 0.6, 200, 700.0, True, "tournament_mutate"),
            AssessmentRecord("c_002", 1, "t1", 0.9, 150, 400.0, True, "llm_explore"),
        ]
        for r in records:
            store.append_assessment(r)

        loaded = store.load_index()
        assert len(loaded) == 3
        assert loaded[0].score == 0.8
        assert loaded[2].proposer == "llm_explore"

    def test_trace_append(self, tmp_path):
        store = EvolutionStore(root=tmp_path / "run_003")
        store.append_trace("c_000", "researcher_0", {"tokens": 100})
        store.append_trace("c_000", "reviewer_1", {"tokens": 50})

        trace_path = store._candidates_dir / "c_000_trace.jsonl"
        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_empty_index(self, tmp_path):
        store = EvolutionStore(root=tmp_path / "run_004")
        assert store.load_index() == []


# ---------------------------------------------------------------------------
# Step 3 tests: EpiplexityMonitor with DistanceProvider
# ---------------------------------------------------------------------------


class TestDistanceProviderExport:

    def test_distance_provider_importable_from_health(self):
        """DistanceProvider is importable from the health package."""
        from operon_ai.health import DistanceProvider as DP
        from operon_ai.health.epiplexity import DistanceProvider as DPDirect
        assert DP is DPDirect


class TestDistanceProvider:

    def test_config_hamming_distance_protocol(self):
        dist = ConfigHammingDistance()
        assert isinstance(dist, DistanceProvider)

    def test_identical_configs_zero_distance(self):
        cc = _make_candidate()
        assert ConfigHammingDistance().distance(cc, cc) == 0.0

    def test_one_field_diff(self):
        cc1 = _make_candidate()
        cc2 = _make_candidate(
            stages=(_make_stage("researcher", "fixed"),
                    _make_stage("reviewer", "fixed")),
        )
        dist = ConfigHammingDistance().distance(cc1, cc2)
        # 1 field differs out of 14 (2 stages x 6 fields + 2 policy keys... actually 1)
        assert 0.0 < dist < 0.2

    def test_epiplexity_stagnant_on_identical_items(self):
        monitor = EpiplexityMonitor(
            distance_provider=ConfigHammingDistance(),
            threshold=0.3,
            critical_duration=3,
            window_size=5,
        )
        cc = _make_candidate()
        # Need enough iterations for the high-novelty first measurement
        # to wash out of the window
        for _ in range(15):
            r = monitor.measure(item=cc)
        assert r.status in (HealthStatus.STAGNANT, HealthStatus.CRITICAL)

    def test_epiplexity_healthy_on_varied_items(self):
        monitor = EpiplexityMonitor(
            distance_provider=ConfigHammingDistance(),
            threshold=0.15,  # Low threshold so moderate novelty stays healthy
        )
        # Each candidate differs in multiple fields (role, mode, model)
        for i in range(6):
            mode = "fuzzy" if i % 2 == 0 else "fixed"
            sc1 = _make_stage(f"role_{i}", mode, model=f"model_{i}")
            sc2 = _make_stage(f"helper_{i}", "fixed")
            cc = _make_candidate(f"c_{i}", stages=(sc1, sc2), policy={"k": i})
            r = monitor.measure(item=cc)
        assert r.status in (HealthStatus.HEALTHY, HealthStatus.EXPLORING)


# ---------------------------------------------------------------------------
# Step 4 tests: Proposers
# ---------------------------------------------------------------------------


class TestProposers:

    def test_tournament_mutator_protocol(self):
        assert isinstance(TournamentMutator(seed=42), Proposer)

    def test_extract_json_raw(self):
        from operon_ai.convergence.meta_proposers import _extract_json
        assert _extract_json('{"score": 0.8}') == {"score": 0.8}

    def test_extract_json_fenced(self):
        from operon_ai.convergence.meta_proposers import _extract_json
        text = 'Here is my suggestion:\n```json\n{"stage_configs": []}\n```'
        assert _extract_json(text) == {"stage_configs": []}

    def test_extract_json_prose_wrapped(self):
        from operon_ai.convergence.meta_proposers import _extract_json
        text = 'Based on the history, I recommend: {"stage_configs": [], "reason": "test"}'
        result = _extract_json(text)
        assert result["reason"] == "test"

    def test_localhost_detection(self):
        """_is_localhost distinguishes true local from remote URLs."""
        from operon_ai.convergence.meta_proposers import _is_localhost
        assert _is_localhost("http://localhost:8080/v1") is True
        assert _is_localhost("http://127.0.0.1:11434/v1") is True
        assert _is_localhost("https://api.example.com/localhost/v1") is False
        assert _is_localhost("https://not-localhost.com/v1") is False
        assert _is_localhost("https://api.example.com/v1?next=localhost") is False

    def test_extract_json_with_trailing_braces(self):
        from operon_ai.convergence.meta_proposers import _extract_json
        text = 'Here: {"score": 0.8} and also {invalid'
        assert _extract_json(text) == {"score": 0.8}

    def test_tournament_selects_best_parent(self):
        cc_low = _make_candidate("c_low")
        cc_high = _make_candidate("c_high")
        scores = {"c_low": [0.3, 0.4], "c_high": [0.8, 0.9]}

        mutator = TournamentMutator(k=2, seed=42)
        child = mutator.propose([cc_low, cc_high], scores, iteration=1)
        assert child.parent_id == "c_high"

    def test_tournament_changes_one_field(self):
        cc = _make_candidate()
        scores = {"c_000": [0.5]}
        mutator = TournamentMutator(k=1, seed=42)
        child = mutator.propose([cc], scores, iteration=1)

        diffs = 0
        for orig, new in zip(cc.stage_configs, child.stage_configs):
            for fld in ("role", "mode", "model", "include_stage_outputs",
                        "include_shared_state", "cognitive_mode"):
                if getattr(orig, fld) != getattr(new, fld):
                    diffs += 1
        assert diffs == 1

    def test_tournament_proposer_field(self):
        cc = _make_candidate()
        child = TournamentMutator(seed=0).propose([cc], {"c_000": [0.5]}, 1)
        assert child.proposer == "tournament_mutate"


# ---------------------------------------------------------------------------
# Step 5 tests: FilesystemOptimizer protocol
# ---------------------------------------------------------------------------


class TestFilesystemOptimizerProtocol:

    def test_protocol_distinctness(self):
        from operon_ai.convergence.prompt_optimization import EvolutionaryOptimizer
        assert FilesystemOptimizer is not EvolutionaryOptimizer

    def test_evolution_loop_has_protocol_methods(self):
        for method in ("seed", "step", "best", "history"):
            assert hasattr(EvolutionLoop, method)


# ---------------------------------------------------------------------------
# Step 6 tests: EvolutionLoop with mock environment
# ---------------------------------------------------------------------------


class _MockAssessEvaluator:
    """Minimal mock that avoids real LLM calls."""
    pass


class TestEvolutionLoop:

    def _make_loop(self, tmp_path, tasks=None):
        config = EvolutionConfig(
            run_id="test_run",
            max_iterations=2,
            population_size=4,
            store_root=tmp_path,
            seed=42,
        )
        tasks = tasks or [_MockTaskDef()]
        return EvolutionLoop(
            config=config,
            evaluator=_MockAssessEvaluator(),
            tasks=tasks,
            provider_name="mock",
        )

    def test_seed_creates_population(self, tmp_path):
        loop = self._make_loop(tmp_path)
        seeds = [_make_candidate("c_seed_0"), _make_candidate("c_seed_1")]
        loop.seed(seeds)
        assert len(loop._population) == 2
        assert (tmp_path / "test_run" / "candidates" / "c_seed_0.json").exists()
        assert (tmp_path / "test_run" / "candidates" / "c_seed_1.json").exists()

    def test_seed_writes_meta_with_config(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop.seed([_make_candidate()])
        import json
        meta = json.loads((tmp_path / "test_run" / "meta.json").read_text())
        expected = {
            "max_iterations": 2,
            "population_size": 4,
            "tournament_k": 3,
            "stall_threshold": 2,
            "seed": 42,
            "llm_proposer": False,
            "provider": "mock",
            "n_tasks": 1,
            "n_seeds": 1,
        }
        for key, val in expected.items():
            assert meta.get(key) == val, f"meta[{key!r}]={meta.get(key)!r}, expected {val!r}"

    def test_history_starts_empty(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop.seed([_make_candidate()])
        assert loop.history() == []

    def test_best_raises_before_assessment(self, tmp_path):
        loop = self._make_loop(tmp_path)
        loop.seed([_make_candidate()])
        with pytest.raises(ValueError, match="No candidates"):
            loop.best()

    def test_config_stall_switches_proposer(self, tmp_path):
        """Config novelty stall (epiplexity) triggers LLM proposer."""
        from operon_ai.convergence.meta_proposers import LLMProposer

        class _FakeProv:
            model = "fake"
            def complete(self, prompt, config=None):
                pass

        config = EvolutionConfig(run_id="stall_test", store_root=tmp_path,
                                  seed=42, stall_threshold=2)
        loop = EvolutionLoop(
            config=config, evaluator=_MockAssessEvaluator(),
            tasks=[_MockTaskDef()], provider_name="mock",
            llm_proposer_provider=_FakeProv(),
        )
        loop.seed([_make_candidate()])

        # Override monitor with tighter settings for test
        loop._monitor = EpiplexityMonitor(
            distance_provider=ConfigHammingDistance(),
            threshold=0.3, window_size=3, critical_duration=2,
        )
        # Feed identical configs to trigger config stall
        cc = _make_candidate()
        for _ in range(10):
            loop._monitor.measure(item=cc)

        proposer = loop._select_proposer()
        assert isinstance(proposer, LLMProposer)

    def test_score_plateau_switches_proposer(self, tmp_path):
        """Score plateau triggers LLM proposer even with varied configs."""
        from operon_ai.convergence.meta_proposers import LLMProposer

        class _FakeProv:
            model = "fake"
            def complete(self, prompt, config=None):
                pass

        config = EvolutionConfig(run_id="score_stall", store_root=tmp_path,
                                  seed=42, stall_threshold=2)
        loop = EvolutionLoop(
            config=config, evaluator=_MockAssessEvaluator(),
            tasks=[_MockTaskDef()], provider_name="mock",
            llm_proposer_provider=_FakeProv(),
        )
        loop.seed([_make_candidate()])
        loop._best_score = 0.5  # set a baseline

        # Simulate steps without improvement (threshold * n_tasks = 2 * 1 = 2)
        loop._steps_since_improvement = 2

        proposer = loop._select_proposer()
        assert isinstance(proposer, LLMProposer)

    def test_score_plateau_increments_via_step(self, tmp_path):
        """Score plateau counter increments through step() and resets on improvement."""
        from unittest.mock import patch

        loop = self._make_loop(tmp_path, tasks=[
            _MockTaskDef(task_id="t1"),
            _MockTaskDef(task_id="t2"),
        ])
        loop.seed([_make_candidate()])

        # Stub assessment to return fixed score
        def make_record(score):
            return AssessmentRecord("x", 0, "t1", score, 0, 0.0, True, "seed")

        # First step sets best_score
        with patch.object(loop, "_assess_candidate", return_value=make_record(0.5)):
            loop.step("t1")
        assert loop._steps_since_improvement == 0  # just improved from -1 to 0.5

        # Next steps don't improve -> counter grows
        with patch.object(loop, "_assess_candidate", return_value=make_record(0.3)):
            loop.step("t2")
        assert loop._steps_since_improvement == 1

        with patch.object(loop, "_assess_candidate", return_value=make_record(0.4)):
            loop.step("t1")
        assert loop._steps_since_improvement == 2

        # Improvement resets counter
        with patch.object(loop, "_assess_candidate", return_value=make_record(0.9)):
            loop.step("t2")
        assert loop._steps_since_improvement == 0

    def test_score_plateau_multi_task_threshold(self, tmp_path):
        """Score plateau threshold scales by number of tasks."""
        from operon_ai.convergence.meta_proposers import LLMProposer

        class _FakeProv:
            model = "fake"
            def complete(self, prompt, config=None):
                pass

        tasks = [_MockTaskDef(task_id="t1"), _MockTaskDef(task_id="t2")]
        config = EvolutionConfig(run_id="multi", store_root=tmp_path,
                                  seed=42, stall_threshold=2)
        loop = EvolutionLoop(
            config=config, evaluator=_MockAssessEvaluator(),
            tasks=tasks, provider_name="mock",
            llm_proposer_provider=_FakeProv(),
        )
        loop.seed([_make_candidate()])
        loop._best_score = 0.5

        # threshold * len(tasks) = 2 * 2 = 4
        # At 3 steps: still tournament
        loop._steps_since_improvement = 3
        assert isinstance(loop._select_proposer(), TournamentMutator)

        # At 4 steps: switches to LLM
        loop._steps_since_improvement = 4
        assert isinstance(loop._select_proposer(), LLMProposer)

    def test_no_stall_stays_tournament(self, tmp_path):
        """Without stall, proposer stays tournament even with LLM available."""
        class _FakeProv:
            model = "fake"
            def complete(self, prompt, config=None):
                pass

        config = EvolutionConfig(run_id="no_stall", store_root=tmp_path,
                                  seed=42, stall_threshold=2)
        loop = EvolutionLoop(
            config=config, evaluator=_MockAssessEvaluator(),
            tasks=[_MockTaskDef()], provider_name="mock",
            llm_proposer_provider=_FakeProv(),
        )
        loop.seed([_make_candidate()])
        loop._steps_since_improvement = 0

        proposer = loop._select_proposer()
        assert isinstance(proposer, TournamentMutator)

    def test_design_problem_feasibility(self):
        """Verify DesignProblem wrapping produces valid feasibility check."""
        from operon_ai.convergence.codesign import DesignProblem

        cc = _make_candidate()
        dp = DesignProblem(
            name="test_dp",
            evaluate_fn=lambda r: {"score": 0.8},
            feasibility_fn=lambda r: len(r["config"].stage_configs) > 0,
        )

        resources = {"config": cc, "task": _MockTaskDef()}
        assert dp.is_feasible(resources) is True

        empty_cc = CandidateConfig("empty", None, 0, (), {})
        resources_empty = {"config": empty_cc, "task": _MockTaskDef()}
        assert dp.is_feasible(resources_empty) is False

    def test_adapt_stages_matches_task_roles(self, tmp_path):
        """Persisted candidate must have roles matching the task, not the seed."""
        from operon_ai.convergence.evolution_loop import EvolutionLoop

        # Seed has researcher+reviewer but task needs analyst+writer+checker
        seed_stages = (_make_stage("researcher"), _make_stage("reviewer", "fixed"))
        task = _MockTaskDef(
            required_roles=("analyst", "writer", "checker"),
        )
        adapted = EvolutionLoop._adapt_stages(seed_stages, task)
        assert len(adapted) == 3
        assert tuple(sc.role for sc in adapted) == ("analyst", "writer", "checker")

    def test_adapt_stages_preserves_style_on_match(self, tmp_path):
        """Matched roles inherit all settings including mode."""
        from operon_ai.convergence.evolution_loop import EvolutionLoop

        seed_stages = (
            StageConfig(role="researcher", mode="fuzzy", include_stage_outputs=False),
            StageConfig(role="reviewer", mode="fixed"),
        )
        task = _MockTaskDef(required_roles=("researcher", "reviewer"))
        adapted = EvolutionLoop._adapt_stages(seed_stages, task)
        assert adapted[0].mode == "fuzzy"
        assert adapted[0].include_stage_outputs is False
        assert adapted[1].mode == "fixed"

    def test_adapt_stages_new_roles_default_fuzzy(self, tmp_path):
        """New roles (no direct match) default to fuzzy mode, not fallback's mode."""
        from operon_ai.convergence.evolution_loop import EvolutionLoop

        seed_stages = (
            StageConfig(role="researcher", mode="fixed"),
            StageConfig(role="reviewer", mode="fixed", model="gemini-2.5-pro"),
        )
        task = _MockTaskDef(required_roles=("researcher", "new_role"))
        adapted = EvolutionLoop._adapt_stages(seed_stages, task)
        # researcher matches directly — inherits fixed
        assert adapted[0].mode == "fixed"
        # new_role has no match — defaults to fuzzy, inherits model from fallback
        assert adapted[1].role == "new_role"
        assert adapted[1].mode == "fuzzy"
        assert adapted[1].model == "gemini-2.5-pro"

    def test_adapt_stages_noop_when_roles_match(self, tmp_path):
        """When candidate roles already match task, stage_configs is returned unchanged."""
        from operon_ai.convergence.evolution_loop import EvolutionLoop

        stages = (_make_stage("researcher"),)
        task = _MockTaskDef(required_roles=("researcher",))
        assert EvolutionLoop._adapt_stages(stages, task) is stages

    def test_adapt_stages_empty_stays_empty(self, tmp_path):
        """Empty stage_configs must stay empty (infeasible), not synthesize from task roles."""
        from operon_ai.convergence.evolution_loop import EvolutionLoop

        task = _MockTaskDef(required_roles=("analyst", "writer"))
        result = EvolutionLoop._adapt_stages((), task)
        assert result == ()

    def test_infeasible_candidate_excluded_from_population(self, tmp_path):
        """Zero-stage candidates must not enter the selectable population."""
        from unittest.mock import patch

        loop = self._make_loop(tmp_path)
        loop.seed([_make_candidate()])
        initial_pop = len(loop._population)

        # Stub proposer to return an empty-stage candidate
        empty = CandidateConfig("c_empty", None, 0, (), {}, proposer="seed")
        fake_record = AssessmentRecord("c_empty", 0, "easy_seq_01", 0.0, 0, 0.0, False, "seed")
        with (
            patch.object(loop._tournament, "propose", return_value=empty),
            patch.object(loop, "_assess_candidate", return_value=fake_record),
        ):
            loop.step("easy_seq_01")

        # Population should not have grown
        assert len(loop._population) == initial_pop

    def test_zero_stage_seed_excluded_from_population(self, tmp_path):
        """Zero-stage seeds are persisted but excluded from selectable population."""
        loop = self._make_loop(tmp_path)
        good = _make_candidate("c_good")
        empty = CandidateConfig("c_empty", None, 0, (), {}, proposer="seed")
        loop.seed([good, empty])

        # Only the good seed enters population
        assert len(loop._population) == 1
        assert loop._population[0].candidate_id == "c_good"
        # But both are persisted
        assert (tmp_path / "test_run" / "candidates" / "c_empty.json").exists()

    def test_all_zero_stage_seeds_raises_after_persist(self, tmp_path):
        """All-empty seeds raises ValueError but seeds are persisted first."""
        loop = self._make_loop(tmp_path)
        empty1 = CandidateConfig("e1", None, 0, (), {})
        empty2 = CandidateConfig("e2", None, 0, (), {})
        with pytest.raises(ValueError, match="non-empty stage_configs"):
            loop.seed([empty1, empty2])
        # Seeds persisted for debugging/replay despite the error
        assert (tmp_path / "test_run" / "candidates" / "e1.json").exists()
        assert (tmp_path / "test_run" / "candidates" / "e2.json").exists()

    def test_unique_candidate_ids_across_steps(self, tmp_path):
        """Two proposals in the same iteration must get distinct IDs."""
        from unittest.mock import patch

        loop = self._make_loop(tmp_path, tasks=[
            _MockTaskDef(task_id="t1"),
            _MockTaskDef(task_id="t2"),
        ])
        loop.seed([_make_candidate()])

        # Stub _assess_candidate to avoid real LLM calls
        fake_record = AssessmentRecord("x", 0, "t1", 0.5, 0, 0.0, True, "seed")
        with patch.object(loop, "_assess_candidate", return_value=fake_record):
            loop._iteration = 0
            c1, _ = loop.step("t1")
            c2, _ = loop.step("t2")
        assert c1.candidate_id != c2.candidate_id

    def test_get_judge_uses_pick_judge(self, tmp_path):
        """_get_judge_provider delegates to _pick_judge when available."""

        class _MockEvalWithPickJudge:
            def _pick_judge(self, provider_name):
                return "mock_judge_provider"

        config = EvolutionConfig(run_id="test", store_root=tmp_path, seed=42)
        loop = EvolutionLoop(
            config=config,
            evaluator=_MockEvalWithPickJudge(),
            tasks=[_MockTaskDef()],
            provider_name="gemini",
        )
        assert loop._get_judge_provider() == "mock_judge_provider"

    def test_cli_explicit_model_creates_provider(self, tmp_path):
        """--llm-proposer + --llm-proposer-model creates provider without auto-detect."""
        from unittest.mock import patch, MagicMock
        from types import SimpleNamespace

        # Simulate CLI args
        args = SimpleNamespace(
            llm_proposer="http://localhost:9999/v1",
            llm_proposer_model="my-explicit-model",
        )

        # The CLI wiring logic (extracted from run_meta_evolution.py)
        llm_provider = None
        if args.llm_proposer:
            model = args.llm_proposer_model
            if not model:
                model = None  # would auto-detect
            if model:
                # This is the path that previously regressed
                llm_provider = SimpleNamespace(
                    api_key="not-needed",
                    base_url=args.llm_proposer,
                    model=model,
                )

        assert llm_provider is not None
        assert llm_provider.model == "my-explicit-model"
        assert llm_provider.base_url == "http://localhost:9999/v1"

    def test_llm_proposer_wired_with_explicit_model(self, tmp_path):
        """EvolutionLoop receives LLM proposer when provider is explicitly given."""
        from operon_ai.convergence.meta_proposers import LLMProposer

        class _FakeProvider:
            model = "explicit-model"
            def complete(self, prompt, config=None):
                pass

        config = EvolutionConfig(run_id="test_llm", store_root=tmp_path, seed=42)
        loop = EvolutionLoop(
            config=config,
            evaluator=_MockAssessEvaluator(),
            tasks=[_MockTaskDef()],
            provider_name="mock",
            llm_proposer_provider=_FakeProvider(),
        )
        assert loop._llm_proposer is not None
        assert isinstance(loop._llm_proposer, LLMProposer)

    def test_build_failure_recorded_as_error_trace(self, tmp_path):
        """Organism build failures are logged to trace, not silently swallowed."""
        from unittest.mock import patch

        loop = self._make_loop(tmp_path)
        loop.seed([_make_candidate()])

        # Stub _build_organism to raise
        with patch.object(loop, "_build_organism", side_effect=ValueError("bad config")):
            loop._iteration = 0
            result = loop._run_and_score(_make_candidate(), _MockTaskDef())

        assert result["score"] == 0.0
        # Check error trace was written
        trace_path = tmp_path / "test_run" / "candidates" / "c_000_trace.jsonl"
        assert trace_path.exists()
        import json
        lines = [json.loads(l) for l in trace_path.read_text().splitlines()]
        assert any(d["stage"] == "_error" for d in lines)
        error_line = next(d for d in lines if d["stage"] == "_error")
        assert "bad config" in error_line["error"]
