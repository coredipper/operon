"""C8 EvolutionLoop: meta-harness for evolving organism configurations.

Thin glue connecting all C8 components: Genome mapping, filesystem store,
EpiplexityMonitor stall detection, and hybrid proposers.  Each evolution
step is wrapped as a DesignProblem to test whether co-design composition
captures evolutionary dynamics.

Implements the FilesystemOptimizer protocol.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..health.epiplexity import EpiplexityMonitor
from ..organelles.nucleus import Nucleus
from ..patterns.organism import skill_organism
from ..patterns.types import SkillStage

from .codesign import DesignProblem
from .meta_proposers import LLMProposer, Proposer, TournamentMutator
from .meta_store import EvolutionStore
from .meta_types import (
    AssessmentRecord,
    CandidateConfig,
    ConfigHammingDistance,
    StageConfig,
    candidate_to_genome,
)


@dataclass
class EvolutionConfig:
    """Configuration for the meta-evolution loop."""

    run_id: str = ""
    max_iterations: int = 20
    population_size: int = 8
    tournament_k: int = 3
    stall_threshold: int = 2  # consecutive stagnant readings to trigger LLM proposer
    store_root: Path = field(default_factory=lambda: Path(".operon/evolution"))
    seed: int | None = None


class EvolutionLoop:
    """Meta-harness: evolve organism configs against benchmark tasks.

    Uses biological primitives where they fit naturally:
    - Genome for candidate representation (gene abstraction test)
    - DesignProblem for evaluation wrapping (co-design composition test)
    - EpiplexityMonitor for stall detection (epistemic health generalization test)
    - Filesystem history for LLM proposer (exogenous signal test)
    """

    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: Any,  # LiveEvaluator or mock
        tasks: list[Any],  # list[TaskDefinition]
        provider_name: str = "gemini",
        llm_proposer_provider: Any | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._tasks = tasks
        self._provider_name = provider_name

        run_id = config.run_id or f"run_{uuid.uuid4().hex[:8]}"
        self._store = EvolutionStore(root=config.store_root / run_id)

        # Population and scores
        self._population: list[CandidateConfig] = []
        self._scores: dict[str, list[float]] = {}
        self._all_records: list[AssessmentRecord] = []
        self._best_score: float = -1.0
        self._best_candidate: CandidateConfig | None = None
        self._iteration: int = 0
        self._step_counter: int = 0  # monotonic, unique per proposal

        # Proposers
        self._tournament = TournamentMutator(
            k=config.tournament_k, seed=config.seed,
        )
        self._llm_proposer: LLMProposer | None = None
        if llm_proposer_provider is not None:
            self._llm_proposer = LLMProposer(
                provider=llm_proposer_provider,
                store=self._store,
                fallback_seed=config.seed,
            )

        # Stall detection: dual signal
        # 1. Config novelty via EpiplexityMonitor (detects mutation churn)
        self._monitor = EpiplexityMonitor(
            distance_provider=ConfigHammingDistance(),
            threshold=0.2,
            critical_duration=5,
        )
        # 2. Score plateau tracker (detects fitness stagnation)
        self._steps_since_improvement: int = 0

    # -- FilesystemOptimizer interface ---------------------------------------

    def seed(self, initial_configs: list[CandidateConfig]) -> None:
        """Initialize the population with seed candidates.

        All seeds are persisted for history, but only feasible (non-empty
        stage) seeds enter the selectable population.
        """
        # Persist all seeds for history first, then validate.
        for cc in initial_configs:
            genome = candidate_to_genome(cc)
            self._store.save_candidate(cc, genome)
            self._scores[cc.candidate_id] = []

        self._population = [cc for cc in initial_configs if cc.stage_configs]
        if not self._population:
            raise ValueError(
                "At least one seed must have non-empty stage_configs"
            )

        self._store.save_meta({
            "run_id": self._store.root.name,
            "max_iterations": self._config.max_iterations,
            "population_size": self._config.population_size,
            "tournament_k": self._config.tournament_k,
            "stall_threshold": self._config.stall_threshold,
            "seed": self._config.seed,
            "provider": self._provider_name,
            "llm_proposer": self._llm_proposer is not None,
            "n_tasks": len(self._tasks),
            "n_seeds": len(initial_configs),
        })

    def step(self, task_id: str) -> tuple[CandidateConfig, AssessmentRecord]:
        """Run one evolution step: propose, evaluate, record."""
        task = next((t for t in self._tasks if t.task_id == task_id), None)
        if task is None:
            raise ValueError(f"Unknown task: {task_id}")

        # Select proposer based on epistemic health
        proposer = self._select_proposer()

        # Propose + adapt — catch errors to prevent run termination
        fallback_used = False
        fallback_reason = ""
        try:
            raw = proposer.propose(
                self._population, self._scores, self._iteration,
            )
            self._step_counter += 1
            adapted_stages, name_map = self._adapt_stages_with_mapping(
                raw.stage_configs, task,
            )
            if name_map:
                new_set = {f"{sc.role}_{i}" for i, sc in enumerate(adapted_stages)}
                adapted_edges = tuple(
                    (name_map[s], name_map[d])
                    for s, d in raw.edges
                    if s in name_map and d in name_map
                    and name_map[s] in new_set and name_map[d] in new_set
                    and name_map[s] != name_map[d]
                )
            else:
                adapted_edges = raw.edges
        except Exception as exc:
            # Fallback: adapt a known-good population member for this task
            self._step_counter += 1
            fallback_used = True
            fallback_reason = f"{type(exc).__name__}: {exc}"
            source = self._population[0]
            adapted_stages, _ = self._adapt_stages_with_mapping(
                source.stage_configs, task,
            )
            adapted_edges = ()  # safe: no edges on fallback
            raw = source  # for intervention_policy/topology

        candidate = CandidateConfig(
            candidate_id=f"c_{self._step_counter:04d}",
            parent_id=raw.candidate_id if fallback_used else raw.parent_id,
            iteration=self._iteration,
            stage_configs=adapted_stages,
            intervention_policy=raw.intervention_policy,
            topology=raw.topology,
            edges=adapted_edges,
            proposer="proposal_fallback" if fallback_used else raw.proposer,
            reason=(fallback_reason if fallback_used else raw.reason),
        )

        # Evaluate via DesignProblem wrapping
        record = self._assess_candidate(candidate, task)

        # Persist
        genome = candidate_to_genome(candidate)
        self._store.save_candidate(candidate, genome)
        self._store.append_assessment(record)
        self._all_records.append(record)

        # Update scores and population — only feasible candidates enter
        # the selectable population to prevent zero-stage candidates from
        # crashing TournamentMutator.
        self._scores.setdefault(candidate.candidate_id, []).append(record.score)
        if record.feasible and candidate.stage_configs:
            self._population.append(candidate)

        # Trim population to size limit (keep best by mean score)
        if len(self._population) > self._config.population_size:
            self._population.sort(
                key=lambda c: (
                    sum(self._scores.get(c.candidate_id, [0.0]))
                    / max(len(self._scores.get(c.candidate_id, [0.0])), 1)
                ),
                reverse=True,
            )
            self._population = self._population[:self._config.population_size]

        # Track best + score plateau
        if record.score > self._best_score:
            self._best_score = record.score
            self._best_candidate = candidate
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        # Feed epiplexity monitor for config novelty stall detection
        self._monitor.measure(item=candidate)

        return candidate, record

    def best(self) -> CandidateConfig:
        """Return the current best candidate."""
        if self._best_candidate is None:
            raise ValueError("No candidates evaluated yet")
        return self._best_candidate

    def history(self) -> list[AssessmentRecord]:
        """Return full evaluation history."""
        return list(self._all_records)

    # -- Main loop -----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the full evolution loop.

        Returns summary dict with best candidate, trajectory, and store path.
        """
        task_cycle = self._tasks * self._config.max_iterations
        task_idx = 0

        for iteration in range(self._config.max_iterations):
            self._iteration = iteration
            for task in self._tasks:
                if task_idx >= len(task_cycle):
                    break
                try:
                    self.step(task.task_id)
                except Exception as run_exc:
                    # Log failure but continue evolution
                    self._store.append_trace(
                        f"step_error_{task_idx}", "_error",
                        {"error": f"{type(run_exc).__name__}: {run_exc}",
                         "task_id": task.task_id,
                         "iteration": iteration},
                    )
                task_idx += 1

        return {
            "best": self._best_candidate,
            "best_score": self._best_score,
            "total_assessments": len(self._all_records),
            "store_path": str(self._store.root),
            "monitor_stats": self._monitor.stats(),
        }

    # -- Internal: evaluation ------------------------------------------------

    def _assess_candidate(
        self,
        config: CandidateConfig,
        task: Any,
    ) -> AssessmentRecord:
        """Evaluate a candidate by building an organism and running it.

        Wraps the evaluation as a DesignProblem to test co-design
        composition at the meta-level.
        """
        # Wrap as DesignProblem: resources -> functionalities
        result: dict[str, Any] = {}

        def evaluate_fn(resources: dict[str, Any]) -> dict[str, Any]:
            return self._run_and_score(resources["config"], resources["task"])

        def feasibility_fn(resources: dict[str, Any]) -> bool:
            return self._validate_topology(resources["config"])

        dp = DesignProblem(
            name=f"assess_{config.candidate_id}_{task.task_id}",
            evaluate_fn=evaluate_fn,
            feasibility_fn=feasibility_fn,
        )

        resources = {"config": config, "task": task}
        feasible = dp.is_feasible(resources)
        if feasible:
            result = dp.evaluate(resources)

        return AssessmentRecord(
            candidate_id=config.candidate_id,
            iteration=self._iteration,
            task_id=task.task_id,
            score=result.get("score", 0.0),
            tokens=result.get("tokens", 0),
            latency_ms=result.get("latency_ms", 0.0),
            feasible=feasible,
            proposer=config.proposer,
        )

    def _run_and_score(
        self,
        config: CandidateConfig,
        task: Any,
    ) -> dict[str, Any]:
        """Build organism from config, run on task, judge quality."""
        try:
            from eval.convergence.live_evaluator import _TASK_PROMPTS
            prompt = _TASK_PROMPTS.get(task.task_id, task.description)
        except ImportError:
            prompt = task.description

        start = time.time()
        try:
            organism = self._build_organism(config, task)
            run_result = organism.run(prompt)
            elapsed_ms = (time.time() - start) * 1000

            total_tokens = sum(sr.tokens_used for sr in run_result.stage_results)
            final_output = str(run_result.final_output or "")

            # Judge quality (lazy import — eval is not in the wheel)
            provider = self._get_judge_provider()
            try:
                from eval.convergence.live_evaluator import _judge_quality
            except ModuleNotFoundError as exc:
                _OPTIONAL = {"eval", "eval.convergence", "eval.convergence.live_evaluator"}
                if exc.name in _OPTIONAL:
                    _judge_quality = None
                else:
                    raise
            if _judge_quality is not None:
                judge_result = _judge_quality(str(prompt), final_output, provider)
                score = judge_result[0]
            else:
                score = 0.5  # no judge available — neutral score

            # Record trace
            for sr in run_result.stage_results:
                self._store.append_trace(config.candidate_id, sr.stage_name, {
                    "role": sr.role,
                    "model": sr.model,
                    "tokens": sr.tokens_used,
                    "latency_ms": sr.latency_ms,
                    "action_type": sr.action_type,
                    "produced_output": sr.output is not None,
                    "output_preview": "" if sr.output is None else str(sr.output)[:200],
                    "run_step": self._step_counter,
                })

            return {
                "score": score,
                "tokens": total_tokens,
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000
            self._store.append_trace(config.candidate_id, "_error", {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": elapsed_ms,
                "run_step": self._step_counter,
            })
            return {"score": 0.0, "tokens": 0, "latency_ms": elapsed_ms}

    @staticmethod
    def _validate_topology(config: CandidateConfig) -> bool:
        """Check that a candidate's topology is well-formed.

        Returns True if valid, False if infeasible:
        - Must have at least one stage
        - All edge endpoints must reference existing stage names
        - No self-loops
        """
        if not config.stage_configs:
            return False
        stage_names = {f"{sc.role}_{i}" for i, sc in enumerate(config.stage_configs)}
        for src, dst in config.edges:
            if src not in stage_names or dst not in stage_names:
                return False
            if src == dst:
                return False
        return True

    @staticmethod
    def _adapt_stages(
        stage_configs: tuple[StageConfig, ...],
        task: Any,
    ) -> tuple[StageConfig, ...]:
        """Adapt stage configs to a task's required roles.

        Returns the (possibly unchanged) stage_configs tuple.
        For edge remapping, use _adapt_stages_with_mapping instead.
        """
        adapted, _ = EvolutionLoop._adapt_stages_with_mapping(stage_configs, task)
        return adapted

    @staticmethod
    def _adapt_stages_with_mapping(
        stage_configs: tuple[StageConfig, ...],
        task: Any,
    ) -> tuple[tuple[StageConfig, ...], dict[str, str]]:
        """Adapt stage configs and return (adapted_stages, name_mapping).

        The name_mapping maps old stage names to new stage names,
        used for consistent edge remapping.
        """
        old_names = [f"{sc.role}_{i}" for i, sc in enumerate(stage_configs)]

        if not stage_configs:
            return stage_configs, {}
        task_roles = getattr(task, "required_roles", None)
        if not task_roles:
            return stage_configs, dict(zip(old_names, old_names))
        if tuple(sc.role for sc in stage_configs) == tuple(task_roles):
            return stage_configs, dict(zip(old_names, old_names))

        from collections import defaultdict
        # Group old stages by role (preserving order for nth-occurrence matching)
        old_configs_by_role: dict[str, list[StageConfig]] = defaultdict(list)
        old_names_by_role: dict[str, list[str]] = defaultdict(list)
        for i, sc in enumerate(stage_configs):
            old_configs_by_role[sc.role].append(sc)
            old_names_by_role[sc.role].append(f"{sc.role}_{i}")
        role_used: dict[str, int] = defaultdict(int)

        fallback = stage_configs[-1] if stage_configs else None
        adapted = []
        name_map: dict[str, str] = {}
        for new_i, role in enumerate(task_roles):
            new_name = f"{role}_{new_i}"
            occ = role_used[role]
            role_used[role] += 1

            # Pick config from the same occurrence used for name mapping
            old_configs = old_configs_by_role.get(role, [])
            old_names = old_names_by_role.get(role, [])
            if occ < len(old_names):
                name_map[old_names[occ]] = new_name

            source = old_configs[occ] if occ < len(old_configs) else None
            if source:
                adapted.append(StageConfig(
                    role=role,
                    mode=source.mode,
                    model=source.model,
                    include_stage_outputs=source.include_stage_outputs,
                    include_shared_state=source.include_shared_state,
                    cognitive_mode=source.cognitive_mode,
                ))
            elif fallback:
                adapted.append(StageConfig(
                    role=role,
                    mode="fuzzy",
                    model=fallback.model,
                    include_stage_outputs=fallback.include_stage_outputs,
                    include_shared_state=fallback.include_shared_state,
                    cognitive_mode=fallback.cognitive_mode,
                ))
            else:
                adapted.append(StageConfig(role=role, mode="fuzzy"))
        return tuple(adapted), name_map

    def _build_organism(self, config: CandidateConfig, task: Any) -> Any:
        """Convert CandidateConfig into a runnable SkillOrganism.

        Assumes config.stage_configs is already adapted to the task's
        roles (done in step() before calling _assess_candidate).
        """
        stages = []
        for i, sc in enumerate(config.stage_configs):
            stages.append(SkillStage(
                name=f"{sc.role}_{i}",
                role=sc.role,
                instructions=(
                    f"You are a {sc.role}. {task.description}\n"
                    f"Process the input and produce output relevant to your role."
                ),
                mode=sc.mode,
                include_stage_outputs=sc.include_stage_outputs,
                include_shared_state=sc.include_shared_state,
            ))

        # Resolve providers for fast/deep nuclei
        fast_provider, deep_provider = self._resolve_providers(config)
        fast_nucleus = Nucleus(provider=fast_provider)
        deep_nucleus = Nucleus(provider=deep_provider)

        return skill_organism(
            stages=stages,
            fast_nucleus=fast_nucleus,
            deep_nucleus=deep_nucleus,
        )

    def _resolve_providers(self, config: CandidateConfig) -> tuple[Any, Any]:
        """Pick fast/deep providers based on stage model overrides or defaults.

        NOTE: SkillOrganism uses one fast nucleus and one deep nucleus for
        the whole organism.  Per-stage model overrides are collapsed to one
        model per tier (last seen wins).  True per-stage model routing
        requires per-stage nucleus aliases, planned for Phase B.
        """
        fast_model = None
        deep_model = None
        for sc in config.stage_configs:
            if sc.model and sc.mode in ("fixed", "fast", "deterministic"):
                fast_model = sc.model
            elif sc.model and sc.mode in ("fuzzy", "deep", "reasoning"):
                deep_model = sc.model

        # Fall back to evaluator's providers
        ev = self._evaluator
        if self._provider_name == "gemini":
            from ..providers.gemini_provider import GeminiProvider
            fast = GeminiProvider(model=fast_model or "gemini-2.5-flash")
            deep = GeminiProvider(model=deep_model or "gemini-2.5-pro")
        elif self._provider_name == "anthropic":
            from ..providers.anthropic_provider import AnthropicProvider
            fast = AnthropicProvider(model=fast_model or "claude-haiku-4-5-20251001")
            deep = AnthropicProvider(model=deep_model or "claude-sonnet-4-6-20260301")
        elif self._provider_name == "openai":
            from ..providers.openai_provider import OpenAIProvider
            fast = OpenAIProvider(model=fast_model or "gpt-5.4-mini")
            deep = OpenAIProvider(model=deep_model or "gpt-5.4")
        elif self._provider_name == "ollama":
            from ..providers.openai_compatible_provider import OpenAICompatibleProvider
            model = getattr(ev, "ollama", None)
            model_name = getattr(model, "model", "deepseek-r1:8b") if model else "deepseek-r1:8b"
            fast = OpenAICompatibleProvider(
                api_key="not-needed", base_url="http://localhost:11434/v1",
                model=fast_model or model_name,
            )
            deep = fast  # Ollama typically has one model
        else:
            # Mock or unknown — try to get providers from evaluator
            fast = getattr(ev, "gemini", None) or getattr(ev, "openai", None)
            deep = fast
        return fast, deep

    def _get_judge_provider(self) -> Any:
        """Get a provider for quality judging.

        Delegates to LiveEvaluator._pick_judge() when available, which
        handles the full fallback chain including local providers (Ollama,
        LM Studio).  Falls back to simple attribute lookup for mocks.
        """
        ev = self._evaluator
        pick = getattr(ev, "_pick_judge", None)
        if pick is not None:
            try:
                return pick(self._provider_name)
            except (ValueError, TypeError):
                pass
        # Fallback for mocks or when _pick_judge is unavailable
        for attr in ("gemini", "anthropic", "openai"):
            prov = getattr(ev, attr, None)
            if prov is not None:
                return prov
        return None

    # -- Internal: proposer selection ----------------------------------------

    def _select_proposer(self) -> Proposer:
        """Choose proposer based on dual stall signals.

        Two independent stall signals, either triggers the LLM proposer:
        1. Config novelty stall (epiplexity): mutation churn without
           structural change — consecutive similar configs
        2. Score plateau: best score hasn't improved in N steps —
           diverse configs but all scoring poorly
        """
        if not self._population or self._llm_proposer is None:
            return self._tournament

        # Signal 1: config novelty stall (epiplexity)
        stats = self._monitor.stats()
        config_stagnant = stats.get("current_consecutive_stagnant", 0)

        # Signal 2: score plateau
        score_stagnant = self._steps_since_improvement

        threshold = self._config.stall_threshold
        if config_stagnant >= threshold or score_stagnant >= threshold * len(self._tasks):
            return self._llm_proposer

        return self._tournament
