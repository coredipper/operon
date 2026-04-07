"""Sequential pipeline validation harness.

Compares epistemic analysis predictions (error bounds, sequential penalty,
tool density) against measured behavior from real LLM execution via
Ollama/Gemma 4.

Scope: SkillOrganism always executes stages as a sequential pipeline,
so this harness validates sequential-topology theorems only. Parallel
speedup is reported as informational.

For each benchmark task × config (guided/unguided) × repeat:
  1. Constructs the same ExternalTopology the live evaluator would build
  2. Runs full ``epistemic.analyze()`` to capture all theorem predictions
  3. Runs the task through ``LiveEvaluator`` with real LLM calls
  4. Pairs predictions with measurements for statistical comparison

Usage::

    from eval.convergence.topology_validation import run_validation
    report = run_validation(n_seeds=3)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from operon_ai.convergence.swarms_adapter import (
    analyze_external_topology,
    build_wiring_diagram,
)
from operon_ai.convergence.types import ExternalTopology
from operon_ai.core.epistemic import analyze as epistemic_analyze
from operon_ai.patterns.advisor import advise_topology

from .live_evaluator import LiveEvaluator, LiveRunResult, _TASK_PROMPTS
from .tasks import TaskDefinition, get_benchmark_tasks


# ---------------------------------------------------------------------------
# Role → Capability mapping
# ---------------------------------------------------------------------------
# Maps task roles to Capability enum values (read_fs, write_fs, net,
# exec_code, money, email_send).  Roles without a clear capability
# mapping are absent — they contribute zero to tool density, which is
# honest (a "synthesizer" doesn't inherently need filesystem or network).

ROLE_CAPABILITIES: dict[str, list[str]] = {
    # Filesystem readers
    "reader": ["read_fs"],
    "parser": ["read_fs"],
    "loader": ["read_fs"],
    "chunker": ["read_fs"],
    "extractor": ["read_fs"],
    "entity_extractor": ["read_fs"],
    "relation_extractor": ["read_fs"],
    "classifier": ["read_fs"],
    "detector": ["read_fs"],
    # Filesystem writers
    "writer": ["write_fs"],
    "logger": ["write_fs"],
    "reporter": ["write_fs"],
    # Filesystem read + write
    "editor": ["read_fs", "write_fs"],
    "formatter": ["read_fs", "write_fs"],
    "merger": ["read_fs", "write_fs"],
    "transformer": ["read_fs", "write_fs"],
    "cleaner": ["read_fs", "write_fs"],
    "graph_builder": ["read_fs", "write_fs"],
    # Code execution
    "runner": ["exec_code"],
    "tester": ["exec_code"],
    "unit_runner": ["exec_code"],
    "integration_runner": ["exec_code"],
    "e2e_runner": ["exec_code"],
    "perf_runner": ["exec_code"],
    "fuzz_runner": ["exec_code"],
    "trainer": ["exec_code"],
    "tuner": ["exec_code"],
    "embedder": ["exec_code"],
    # Code + filesystem
    "linter": ["read_fs", "exec_code"],
    "bug_finder": ["read_fs", "exec_code"],
    "style_checker": ["read_fs", "exec_code"],
    "sast_scanner": ["read_fs", "exec_code"],
    "secret_scanner": ["read_fs", "exec_code"],
    "dep_scanner": ["read_fs", "exec_code"],
    "compliance_checker": ["read_fs", "exec_code"],
    # Network
    "researcher": ["net"],
    "researcher_1": ["net"],
    "researcher_2": ["net"],
    "researcher_3": ["net"],
    "researcher_4": ["net"],
    "researcher_5": ["net"],
    "collector": ["net"],
    "retriever": ["net"],
    "ingester": ["net"],
    "monitor": ["net"],
    # Network + code
    "dast_scanner": ["net", "exec_code"],
    # Network + filesystem
    "backend_dev": ["read_fs", "write_fs", "net"],
    "frontend_dev": ["read_fs", "write_fs", "net"],
    "devops": ["read_fs", "write_fs", "net", "exec_code"],
    "deployer": ["write_fs", "net", "exec_code"],
    "architect": ["read_fs", "net"],
    # Multi-modal processors
    "text_processor": ["read_fs"],
    "image_processor": ["read_fs", "exec_code"],
    "audio_processor": ["read_fs", "exec_code"],
    "video_processor": ["read_fs", "exec_code"],
    "fusion_engine": ["read_fs", "exec_code"],
    # Analysis roles with clear external tool needs
    "feature_engineer": ["read_fs", "exec_code"],
    "security_auditor": ["read_fs", "net"],
    # Pure reasoning / coordination roles — intentionally unmapped so
    # planning_cost_ratio distinguishes tool-heavy from tool-light
    # pipelines rather than collapsing to stage count.
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRecord:
    """All epistemic predictions for a single task-config pair."""

    task_id: str
    config_name: str  # "guided" or "unguided"
    repeat: int  # uncontrolled repetition index (not a deterministic seed)
    # Topology classification
    topology_class: str
    chain_length: int
    parallelism_width: int
    # Theorem 1: Error amplification
    error_n_agents: int
    error_independent_bound: int
    error_centralized_bound: float
    error_detection_rate: float
    error_amplification_ratio: float
    # Theorem 2: Sequential penalty
    seq_chain_length: int
    seq_num_handoffs: int
    seq_overhead_ratio: float
    # Theorem 3: Parallel speedup
    predicted_speedup: float
    num_subtasks: int
    # Theorem 4: Tool density
    total_tools: int
    num_tool_modules: int
    planning_cost_ratio: float
    remote_fraction: float
    # Composite
    risk_score: float


@dataclass(frozen=True)
class MeasurementRecord:
    """All measurements from a single live execution."""

    task_id: str
    config_name: str
    repeat: int  # uncontrolled repetition index (not a deterministic seed)
    success: bool
    quality_score: float
    quality_reason: str
    total_tokens: int
    total_latency_ms: float
    stage_count: int
    per_stage_tokens: tuple[int, ...]
    per_stage_latency_ms: tuple[float, ...]


@dataclass(frozen=True)
class ValidationPair:
    """Matched prediction + measurement for one task-config-repeat triple."""

    prediction: PredictionRecord
    measurement: MeasurementRecord


@dataclass
class TheoremValidation:
    """Correlation results for one epistemic theorem."""

    theorem_name: str
    description: str
    n_pairs: int
    predicted_values: list[float]
    measured_values: list[float]
    spearman_rho: float
    spearman_p: float
    direction_correct: bool
    validation_pass: bool
    informational: bool  # True = not counted toward theorems_validated
    notes: str


@dataclass
class ValidationReport:
    """Complete validation report."""

    pairs: list[ValidationPair]
    theorems: list[TheoremValidation]
    by_topology_class: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Pure-Python Spearman rank correlation
# ---------------------------------------------------------------------------


def _rank(values: list[float]) -> list[float]:
    """Assign average ranks to values (handles ties)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rank correlation with approximate p-value.

    Returns (rho, p_value).  For n < 3 returns (0.0, 1.0).
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0, 1.0

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_rx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_ry) ** 2 for b in ry))

    if den_x == 0 or den_y == 0:
        return 0.0, 1.0

    rho = num / (den_x * den_y)

    # Approximate p-value via t-distribution
    if abs(rho) >= 1.0:
        return rho, 0.0
    t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    # Normal approximation is only reliable for n >= 10; for smaller
    # samples, cap p at a floor to avoid false significance claims.
    p = 2.0 * _norm_sf(abs(t_stat))
    if n < 10:
        p = max(p, 0.1)
    return rho, p


def _norm_sf(z: float) -> float:
    """Survival function of standard normal (1 - CDF), approximation."""
    # Abramowitz & Stegun 26.2.17
    a1, a2, a3 = 0.4361836, -0.1201676, 0.9372980
    t = 1.0 / (1.0 + 0.33267 * abs(z))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    sf = pdf * t * (a1 + t * (a2 + t * a3))
    return sf if z >= 0 else 1.0 - sf


# ---------------------------------------------------------------------------
# Prediction construction (mirrors live_evaluator.py:641-698)
# ---------------------------------------------------------------------------


def build_prediction(
    task: TaskDefinition,
    guided: bool,
    repeat: int = 0,
) -> PredictionRecord:
    """Construct predictions from epistemic analysis without any LLM calls.

    Mirrors the topology construction in ``LiveEvaluator.evaluate_task()``
    (building stages, edges, ExternalTopology) then runs
    ``epistemic.analyze()`` on the resulting WiringDiagram.
    """
    # Guided advice
    advice = None
    if guided:
        advice = advise_topology(
            task_shape=task.task_shape,
            tool_count=task.tool_count,
            subtask_count=task.subtask_count,
            error_tolerance=0.01,
        )

    # Build stage specs (same logic as evaluate_task lines 651-674)
    agent_specs = []
    n_roles = len(task.required_roles)
    for i, role in enumerate(task.required_roles):
        if guided and advice:
            rec = advice.recommended_pattern
            if "reviewer" in rec or "gate" in rec:
                mode = "fixed" if i == n_roles - 1 else "fuzzy"
            else:
                mode = "fuzzy"
        else:
            mode = "fuzzy"
        agent_specs.append({
            "name": f"{role}_{i}",
            "role": role,
            "mode": mode,
            "capabilities": ROLE_CAPABILITIES.get(role, []),
        })

    # Sequential edges (same as evaluate_task lines 686-688)
    realized_edges = [
        (agent_specs[i]["name"], agent_specs[i + 1]["name"])
        for i in range(len(agent_specs) - 1)
    ]

    topology = ExternalTopology(
        source="operon",
        pattern_name=advice.recommended_pattern if advice else "generic_pipeline",
        agents=tuple(agent_specs),
        edges=tuple(realized_edges),
        metadata={"guided": guided, "task_shape": task.task_shape},
    )

    # Run full epistemic analysis on the wiring diagram
    diagram = build_wiring_diagram(topology)
    analysis = epistemic_analyze(diagram)

    # Also get composite risk score from the adapter
    adapter_result = analyze_external_topology(topology)

    return PredictionRecord(
        task_id=task.task_id,
        config_name="guided" if guided else "unguided",
        repeat=repeat,
        topology_class=analysis.classification.topology_class.value,
        chain_length=analysis.classification.chain_length,
        parallelism_width=analysis.classification.parallelism_width,
        error_n_agents=analysis.error_bound.n_agents,
        error_independent_bound=analysis.error_bound.independent_bound,
        error_centralized_bound=analysis.error_bound.centralized_bound,
        error_detection_rate=analysis.error_bound.detection_rate,
        error_amplification_ratio=analysis.error_bound.amplification_ratio,
        seq_chain_length=analysis.sequential.chain_length,
        seq_num_handoffs=analysis.sequential.num_handoffs,
        seq_overhead_ratio=analysis.sequential.overhead_ratio,
        predicted_speedup=analysis.speedup.speedup,
        num_subtasks=analysis.speedup.num_subtasks,
        total_tools=analysis.density.total_tools,
        num_tool_modules=analysis.density.num_modules,
        planning_cost_ratio=analysis.density.planning_cost_ratio,
        remote_fraction=analysis.density.remote_fraction,
        risk_score=adapter_result.risk_score,
    )


# ---------------------------------------------------------------------------
# Measurement extraction
# ---------------------------------------------------------------------------


def _extract_measurement(
    result: LiveRunResult,
    repeat: int,
) -> MeasurementRecord:
    """Extract a MeasurementRecord from a LiveRunResult."""
    per_stage_tokens = tuple(d.get("tokens", 0) for d in result.stage_details)
    per_stage_latency = tuple(d.get("latency_ms", 0.0) for d in result.stage_details)

    return MeasurementRecord(
        task_id=result.task_id,
        config_name=result.config_name,
        repeat=repeat,
        success=result.success,
        quality_score=result.quality_score,
        quality_reason=result.quality_reason,
        total_tokens=result.total_tokens,
        total_latency_ms=result.total_latency_ms,
        stage_count=result.stage_count,
        per_stage_tokens=per_stage_tokens,
        per_stage_latency_ms=per_stage_latency,
    )


# ---------------------------------------------------------------------------
# Single task execution
# ---------------------------------------------------------------------------


def run_single_task(
    evaluator: LiveEvaluator,
    task: TaskDefinition,
    guided: bool,
    repeat: int,
) -> ValidationPair:
    """Run one task, return matched prediction + measurement."""
    prediction = build_prediction(task, guided, repeat)

    result = evaluator.evaluate_task(
        task,
        guided=guided,
        provider_name="ollama",
    )

    measurement = _extract_measurement(result, repeat)
    return ValidationPair(prediction=prediction, measurement=measurement)


# ---------------------------------------------------------------------------
# Theorem validation (statistical comparison)
# ---------------------------------------------------------------------------


def compute_theorem_validations(
    pairs: list[ValidationPair],
) -> list[TheoremValidation]:
    """Compute correlations between predictions and measurements.

    Only pairs where the organism actually executed (``stage_count > 0``)
    are included.  Low-quality runs are kept — they are exactly the data
    points the structural predictors should explain.
    """
    validations = []

    # Exclude infrastructure failures (no stage data) but keep low-quality
    # runs — those are the data points structural predictors should explain.
    valid = [p for p in pairs if p.measurement.per_stage_latency_ms]

    all_pred = [p.prediction for p in valid]
    all_meas = [p.measurement for p in valid]

    # -- Theorem 1: Error Amplification --
    # For sequential pipelines, independent_bound (= n_agents) is the
    # appropriate predictor — each stage can independently introduce errors.
    pred_err = [float(p.error_independent_bound) for p in all_pred]
    meas_err = [1.0 - m.quality_score for m in all_meas]
    rho, p = spearman_rho(pred_err, meas_err)
    validations.append(TheoremValidation(
        theorem_name="ErrorAmplification",
        description="independent_bound (n_agents) vs (1 - quality_score)",
        n_pairs=len(valid),
        predicted_values=pred_err,
        measured_values=meas_err,
        spearman_rho=rho,
        spearman_p=p,
        direction_correct=rho >= 0,
        validation_pass=rho > 0 and p < 0.1,
        informational=False,
        notes="Positive rho = more agents correlates with lower quality",
    ))

    # -- Theorem 2: Sequential Penalty --
    # Higher overhead_ratio → higher mean stage latency
    pred_seq = [p.seq_overhead_ratio for p in all_pred]
    meas_seq = [
        (sum(m.per_stage_latency_ms) / len(m.per_stage_latency_ms))
        if m.per_stage_latency_ms else 0.0
        for m in all_meas
    ]
    rho, p = spearman_rho(pred_seq, meas_seq)
    validations.append(TheoremValidation(
        theorem_name="SequentialPenalty",
        description="overhead_ratio vs mean stage latency (ms)",
        n_pairs=len(valid),
        predicted_values=pred_seq,
        measured_values=meas_seq,
        spearman_rho=rho,
        spearman_p=p,
        direction_correct=rho >= 0,
        validation_pass=rho > 0 and p < 0.1,
        informational=False,
        notes="Positive rho = more overhead predicts more latency per stage",
    ))

    # -- Theorem 3: Parallel Speedup (informational) --
    # SkillOrganism always executes sequentially, so predicted_speedup is
    # constant (~1.0) and no meaningful correlation can be computed.
    # Reported for transparency but not counted toward theorems_validated.
    pred_spd = [p.predicted_speedup for p in all_pred]
    # For a sequential pipeline, no work overlaps — realized speedup is
    # always 1.0.  (sum/max would incorrectly report > 1 for multi-stage
    # chains even when all stages execute serially.)
    meas_spd = [1.0] * len(all_meas)
    rho, p = spearman_rho(pred_spd, meas_spd)
    validations.append(TheoremValidation(
        theorem_name="ParallelSpeedup",
        description="predicted_speedup vs realized parallelism ratio",
        n_pairs=len(valid),
        predicted_values=pred_spd,
        measured_values=meas_spd,
        spearman_rho=rho,
        spearman_p=p,
        direction_correct=rho >= 0,
        validation_pass=False,
        informational=True,
        notes=(
            "Informational only — SkillOrganism is always sequential, so "
            "predicted speedup is constant (~1.0) and untestable."
        ),
    ))

    # -- Theorem 4: Tool Density --
    # Higher planning_cost_ratio → more tokens per stage
    pred_den = [p.planning_cost_ratio for p in all_pred]
    meas_den = [
        m.total_tokens / max(m.stage_count, 1) for m in all_meas
    ]
    rho, p = spearman_rho(pred_den, meas_den)
    validations.append(TheoremValidation(
        theorem_name="ToolDensity",
        description="planning_cost_ratio vs tokens per stage",
        n_pairs=len(valid),
        predicted_values=pred_den,
        measured_values=meas_den,
        spearman_rho=rho,
        spearman_p=p,
        direction_correct=rho >= 0,
        validation_pass=rho > 0 and p < 0.1,
        informational=True,
        notes=(
            "Informational — capabilities are annotated in predictions "
            "via ROLE_CAPABILITIES but not in the live executor's "
            "topology. Reported to check whether role-implied tool "
            "density correlates with token consumption."
        ),
    ))

    # -- Composite Risk Score --
    pred_risk = [p.risk_score for p in all_pred]
    meas_risk = [1.0 - m.quality_score for m in all_meas]
    rho, p = spearman_rho(pred_risk, meas_risk)
    validations.append(TheoremValidation(
        theorem_name="CompositeRisk",
        description="risk_score vs (1 - quality_score)",
        n_pairs=len(valid),
        predicted_values=pred_risk,
        measured_values=meas_risk,
        spearman_rho=rho,
        spearman_p=p,
        direction_correct=rho >= 0,
        validation_pass=rho > 0 and p < 0.1,
        informational=False,
        notes="Positive rho = higher structural risk predicts lower quality",
    ))

    return validations


def _group_by_topology(
    pairs: list[ValidationPair],
) -> dict[str, dict[str, Any]]:
    """Aggregate measurements by topology class."""
    groups: dict[str, list[ValidationPair]] = {}
    for pair in pairs:
        cls = pair.prediction.topology_class
        groups.setdefault(cls, []).append(pair)

    result = {}
    for cls, group in sorted(groups.items()):
        scores = [p.measurement.quality_score for p in group]
        successes = [p.measurement.success for p in group]
        latencies = [p.measurement.total_latency_ms for p in group]
        tokens = [p.measurement.total_tokens for p in group]
        result[cls] = {
            "n_runs": len(group),
            "mean_quality": sum(scores) / len(scores) if scores else 0,
            "success_rate": sum(successes) / len(successes) if successes else 0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "mean_tokens": sum(tokens) / len(tokens) if tokens else 0,
        }
    return result


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _pair_to_dict(pair: ValidationPair) -> dict:
    return {
        "prediction": asdict(pair.prediction),
        "measurement": asdict(pair.measurement),
    }


def _pair_from_dict(d: dict) -> ValidationPair:
    pred = PredictionRecord(**d["prediction"])
    meas_data = dict(d["measurement"])
    meas_data["per_stage_tokens"] = tuple(meas_data["per_stage_tokens"])
    meas_data["per_stage_latency_ms"] = tuple(meas_data["per_stage_latency_ms"])
    meas = MeasurementRecord(**meas_data)
    return ValidationPair(prediction=pred, measurement=meas)


_CHECKPOINT_VERSION = 3  # bump when prediction schema changes


def _save_checkpoint(
    path: str,
    pairs: list[ValidationPair],
    model: str = "",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": _CHECKPOINT_VERSION,
                "model": model,
                "pairs": [_pair_to_dict(p) for p in pairs],
            },
            f,
            indent=2,
        )


def _load_checkpoint(
    path: str,
    *,
    expected_model: str = "",
    valid_keys: set[tuple[str, str, int]] | None = None,
) -> list[ValidationPair]:
    """Load checkpoint, filtering to current task/config/repeat universe.

    If *expected_model* is provided and the checkpoint was saved under a
    different model, the checkpoint is discarded (returns empty list).
    If *valid_keys* is provided, only pairs whose ``(task_id, config, repeat)``
    appear in that set are kept.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Version or model mismatch → discard checkpoint
    if data.get("version") != _CHECKPOINT_VERSION:
        return []
    saved_model = data.get("model", "")
    if expected_model and (not saved_model or saved_model != expected_model):
        return []

    pairs = [_pair_from_dict(d) for d in data.get("pairs", [])]

    # Filter to current universe
    if valid_keys is not None:
        pairs = [
            p for p in pairs
            if (p.prediction.task_id, p.prediction.config_name,
                p.prediction.repeat) in valid_keys
        ]

    return pairs


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_validation(
    n_seeds: int = 3,
    tasks: list[TaskDefinition] | None = None,
    resume: bool = False,
    out_dir: str = "eval/results/topology_validation",
    progress_callback: Any = None,
) -> ValidationReport:
    """Run the full topology validation suite.

    Args:
        n_seeds: Number of repetitions per task-config pair.
        tasks: Subset of tasks to run (default: all 20 benchmark tasks).
        resume: If True, load existing checkpoint and skip completed pairs.
        out_dir: Directory for checkpoint and results.
        progress_callback: Optional ``callable(done, total, task_id, config)``
            invoked after each task completes.

    Returns:
        A :class:`ValidationReport` with all predictions, measurements,
        and per-theorem correlation results.
    """
    if tasks is None:
        tasks = get_benchmark_tasks()

    # Filter to tasks that have prompts
    tasks = [t for t in tasks if t.task_id in _TASK_PROMPTS]

    checkpoint_path = os.path.join(out_dir, "checkpoint.json")

    evaluator = LiveEvaluator()

    # Fail fast if Ollama is not available
    if not evaluator.ollama:
        raise RuntimeError(
            "Ollama not available at localhost:11434 with a chat-capable model. "
            "Start Ollama and pull a model (e.g. ollama pull gemma4)."
        )

    ollama_model = evaluator.ollama.model

    # Build the valid (task_id, config, seed) universe for this run
    valid_keys = {
        (t.task_id, cfg, s)
        for t in tasks
        for cfg in ("unguided", "guided")
        for s in range(n_seeds)
    }

    # Resume or start fresh — filter to current universe and model
    if resume:
        pairs = _load_checkpoint(
            checkpoint_path,
            expected_model=ollama_model,
            valid_keys=valid_keys,
        )
    else:
        pairs = []

    # Only treat pairs with actual stage data as completed — pairs
    # without stage data (infrastructure failures) get retried.
    completed = {
        (p.prediction.task_id, p.prediction.config_name, p.prediction.repeat)
        for p in pairs
        if p.measurement.per_stage_latency_ms
    }
    # Drop failed pairs so they get re-executed
    pairs = [p for p in pairs if p.measurement.per_stage_latency_ms]

    total = len(tasks) * 2 * n_seeds
    done = len(pairs)

    start_time = time.time()

    for task in tasks:
        for guided in [False, True]:
            config_name = "guided" if guided else "unguided"
            for rep in range(n_seeds):
                key = (task.task_id, config_name, rep)
                if key in completed:
                    continue

                try:
                    pair = run_single_task(evaluator, task, guided, rep)
                    pairs.append(pair)
                except Exception as e:
                    # Record failure as zero-quality measurement
                    prediction = build_prediction(task, guided, rep)
                    measurement = MeasurementRecord(
                        task_id=task.task_id,
                        config_name=config_name,
                        repeat=rep,
                        success=False,
                        quality_score=0.0,
                        quality_reason=f"Error: {e}",
                        total_tokens=0,
                        total_latency_ms=0.0,
                        stage_count=0,
                        per_stage_tokens=(),
                        per_stage_latency_ms=(),
                    )
                    pairs.append(ValidationPair(
                        prediction=prediction,
                        measurement=measurement,
                    ))

                done += 1
                _save_checkpoint(checkpoint_path, pairs, model=ollama_model)

                if progress_callback:
                    progress_callback(done, total, task.task_id, config_name)

    elapsed = time.time() - start_time

    # Compute validations
    theorems = compute_theorem_validations(pairs)
    by_class = _group_by_topology(pairs)

    testable = [t for t in theorems if not t.informational]
    n_pass = sum(1 for t in testable if t.validation_pass)
    n_total = len(testable)

    summary = {
        "theorems_validated": n_pass,
        "theorems_total": n_total,
        "total_pairs": len(pairs),
        "success_rate": (
            sum(p.measurement.success for p in pairs) / len(pairs)
            if pairs else 0
        ),
        "mean_quality": (
            sum(p.measurement.quality_score for p in pairs) / len(pairs)
            if pairs else 0
        ),
    }

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": ollama_model,
        "provider": "ollama",
        "n_repeats": n_seeds,
        "n_tasks": len(tasks),
        "n_configs": 2,
        "single_model_note": (
            "For single-model providers (Ollama), guided and unguided "
            "configs use the same model. Predictions are structurally "
            "identical; measurements may differ due to mode assignment."
        ),
        "total_runs": len(pairs),
        "elapsed_seconds": round(elapsed, 1),
    }

    return ValidationReport(
        pairs=pairs,
        theorems=theorems,
        by_topology_class=by_class,
        summary=summary,
        metadata=metadata,
    )
