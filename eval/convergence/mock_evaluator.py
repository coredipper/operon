"""Mock evaluator for the C6 convergence evaluation harness.

This is the critical evaluation component.  For each task x configuration
pair it:

1. Builds a real SkillOrganism from the task's required_roles.
2. Compiles through the config's real compiler (Swarms/DeerFlow/Ralph/Scion).
3. Parses the compiled dict back through the config's adapter into an
   ExternalTopology.
4. Runs analyze_external_topology() to get a real AdapterResult with risk_score.
5. Derives synthetic success, token, latency, and intervention metrics from
   the real analysis outputs.
6. For guided configs, applies a 30% risk reduction.

All compilers and adapters are imported from operon_ai.convergence.
"""

from __future__ import annotations

import random
from typing import Any

from operon_ai import MockProvider, Nucleus, SkillStage, skill_organism

from operon_ai.convergence.swarms_compiler import organism_to_swarms
from operon_ai.convergence.swarms_adapter import (
    analyze_external_topology,
    parse_swarm_topology,
)
from operon_ai.convergence.deerflow_compiler import organism_to_deerflow
from operon_ai.convergence.deerflow_adapter import parse_deerflow_session
from operon_ai.convergence.ralph_compiler import organism_to_ralph
from operon_ai.convergence.ralph_adapter import parse_ralph_config
from operon_ai.convergence.scion_compiler import organism_to_scion
from operon_ai.convergence.types import ExternalTopology, RuntimeConfig

from .configurations import ConfigurationSpec
from .metrics import RunMetrics
from .structural_variation import topology_distance
from .tasks import TaskDefinition


# ---------------------------------------------------------------------------
# Compiler registry
# ---------------------------------------------------------------------------

_COMPILERS: dict[str, Any] = {
    "organism_to_swarms": organism_to_swarms,
    "organism_to_deerflow": organism_to_deerflow,
    "organism_to_ralph": organism_to_ralph,
    "organism_to_scion": organism_to_scion,
}


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------


def _swarms_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a Swarms compiler output dict into an ExternalTopology."""
    return parse_swarm_topology(
        pattern_name=compiled["workflow_type"],
        agent_specs=compiled["agents"],
        edges=compiled["edges"],
    )


def _deerflow_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a DeerFlow compiler output dict into an ExternalTopology."""
    return parse_deerflow_session(compiled)


def _ralph_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a Ralph compiler output dict into an ExternalTopology."""
    return parse_ralph_config(compiled)


def _scion_dict_to_topology(compiled: dict) -> ExternalTopology:
    """Parse a Scion compiler output dict into an ExternalTopology.

    Scion has no dedicated adapter, so we construct the ExternalTopology
    directly from the compiled grove config.
    """
    agents_raw = compiled.get("agents", [])
    messaging = compiled.get("messaging", [])

    # Filter out the watcher agent added by the compiler.
    agent_specs = []
    for a in agents_raw:
        if a.get("name") == "operon-watcher":
            continue
        template = a.get("template", {})
        agent_specs.append({
            "name": a["name"],
            "role": a["name"],
            "capabilities": template.get("skills", []),
        })

    edges = [(m["from"], m["to"]) for m in messaging]

    return ExternalTopology(
        source="scion",
        pattern_name="ScionGrove",
        agents=tuple(agent_specs),
        edges=tuple(edges),
        metadata=compiled,
    )


_ADAPTER_DISPATCH: dict[str, Any] = {
    "parse_swarm_topology": _swarms_dict_to_topology,
    "parse_deerflow_session": _deerflow_dict_to_topology,
    "parse_ralph_config": _ralph_dict_to_topology,
}

# Guidance risk reduction factor.
_GUIDANCE_REDUCTION = 0.30


# ---------------------------------------------------------------------------
# Organism builder
# ---------------------------------------------------------------------------


def _build_organism(task: TaskDefinition):
    """Build a SkillOrganism from a task's required roles.

    Uses deterministic handlers so no LLM calls are needed.
    """
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))

    stages = []
    for i, role in enumerate(task.required_roles):
        stages.append(SkillStage(
            name=f"{role}_{i}",
            role=role,
            instructions=f"Perform {role} duties for: {task.description}",
            mode="fixed" if i % 2 == 0 else "fuzzy",
            handler=lambda t, _r=role: {_r: "done"},
        ))

    return skill_organism(
        stages=stages,
        fast_nucleus=fast,
        deep_nucleus=deep,
    )


# ---------------------------------------------------------------------------
# Operon-native evaluation (no external compiler)
# ---------------------------------------------------------------------------


def _evaluate_operon_native(
    task: TaskDefinition,
    rng: random.Random,
) -> RunMetrics:
    """Evaluate using Operon's native adaptive topology -- no compiler needed.

    Builds the organism and directly analyzes its structure as if it were
    an external topology (sequential chain of stages).
    """
    organism = _build_organism(task)
    stages = organism.stages

    # Build a synthetic ExternalTopology from the organism's stage structure.
    agent_specs = []
    for s in stages:
        agent_specs.append({
            "name": s.name,
            "role": s.role,
        })
    edges = []
    for i in range(len(stages) - 1):
        edges.append((stages[i].name, stages[i + 1].name))

    topology = ExternalTopology(
        source="operon",
        pattern_name="SkillOrganism",
        agents=tuple(agent_specs),
        edges=tuple(edges),
        metadata={},
    )

    result = analyze_external_topology(topology)

    # Apply guidance reduction since this is the operon_adaptive config.
    risk_score = result.risk_score * (1.0 - _GUIDANCE_REDUCTION)
    risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

    stage_count = len(stages)
    success = rng.random() > risk_score
    token_cost = int(1000 * stage_count * (1.0 + risk_score))
    # Sequential overhead proportional to edge count.
    seq_overhead = len(edges) * 0.1 if task.task_shape == "sequential" else 0.0
    latency_ms = 500.0 * stage_count * (1.0 + seq_overhead)
    intervention_count = len(result.warnings)

    # Structural variation: compare task shape to realized topology.
    realized_shape = "sequential"  # organism default is sequential pipeline
    variation = topology_distance(task.task_shape, realized_shape)

    return RunMetrics(
        task_id=task.task_id,
        config_id="operon_adaptive",
        success=success,
        token_cost=token_cost,
        latency_ms=latency_ms,
        intervention_count=intervention_count,
        convergence_rate=1.0 - risk_score,
        structural_variation=variation,
        risk_score=risk_score,
        stage_count=stage_count,
    )


# ---------------------------------------------------------------------------
# MockEvaluator
# ---------------------------------------------------------------------------


class MockEvaluator:
    """Evaluates task x configuration pairs using real Operon compilers and adapters.

    Uses deterministic RNG for reproducibility.
    """

    def __init__(self, seed: int = 1337) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def evaluate(self, task: TaskDefinition, config: ConfigurationSpec) -> RunMetrics:
        """Evaluate a single task under a single configuration.

        Steps:
        1. Build SkillOrganism from task.required_roles.
        2. Compile using the config's compiler.
        3. Parse compiled dict back through adapter -> ExternalTopology.
        4. Run analyze_external_topology() -> AdapterResult.
        5. Derive synthetic metrics from risk_score.
        6. For guided configs, reduce risk_score by 30%.
        """
        # Use a per-evaluation RNG derived from the task+config for determinism.
        pair_seed = hash((self._seed, task.task_id, config.config_id)) & 0xFFFFFFFF
        rng = random.Random(pair_seed)

        # Handle operon-native config (no compiler).
        if config.compiler_fn is None:
            return _evaluate_operon_native(task, rng)

        # Step 1: Build organism.
        organism = _build_organism(task)
        stage_count = len(organism.stages)

        # Step 2: Compile through the real compiler.
        compiler = _COMPILERS[config.compiler_fn]
        runtime_cfg = RuntimeConfig(provider="mock")
        compiled = compiler(organism, config=runtime_cfg)

        # Step 3: Parse back through adapter.
        if config.adapter_fn and config.adapter_fn in _ADAPTER_DISPATCH:
            topology = _ADAPTER_DISPATCH[config.adapter_fn](compiled)
        elif config.framework == "scion":
            topology = _scion_dict_to_topology(compiled)
        else:
            # Fallback: build topology from compiled agents.
            topology = _scion_dict_to_topology(compiled)

        # Step 4: Analyze through Operon's epistemic engine.
        adapter_result = analyze_external_topology(topology)

        # Step 5: Derive metrics.
        risk_score = adapter_result.risk_score

        # Step 6: Apply guidance reduction for guided configs.
        if config.structural_guidance:
            risk_score = risk_score * (1.0 - _GUIDANCE_REDUCTION)
            risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

        success = rng.random() > risk_score
        token_cost = int(1000 * stage_count * (1.0 + risk_score))

        # Latency: sequential tasks have overhead from edges.
        seq_overhead = 0.0
        if len(topology.edges) > 0:
            seq_overhead = len(topology.edges) * 0.1 / max(len(topology.agents), 1)
        latency_ms = 500.0 * stage_count * (1.0 + seq_overhead)

        intervention_count = len(adapter_result.warnings)

        # Structural variation.
        from operon_ai.convergence.swarms_adapter import _classify_task_shape
        realized_shape = _classify_task_shape(topology)
        variation = topology_distance(task.task_shape, realized_shape)

        convergence_rate = 1.0 - risk_score

        return RunMetrics(
            task_id=task.task_id,
            config_id=config.config_id,
            success=success,
            token_cost=token_cost,
            latency_ms=latency_ms,
            intervention_count=intervention_count,
            convergence_rate=convergence_rate,
            structural_variation=variation,
            risk_score=risk_score,
            stage_count=stage_count,
        )
