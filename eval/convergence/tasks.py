"""Benchmark task definitions for the C6 convergence evaluation harness.

Provides 21 tasks across three difficulty levels and three task shapes:
  - 5 easy sequential tasks
  - 8 medium mixed tasks
  - 8 hard parallel tasks

Each task specifies required roles, tool counts, and subtask counts that
drive organism construction in the mock evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

from operon_ai.patterns.repository import TaskFingerprint


@dataclass(frozen=True)
class TaskDefinition:
    """A benchmark task for convergence evaluation."""

    task_id: str
    name: str
    description: str
    difficulty: str  # easy / medium / hard
    task_shape: str  # sequential / parallel / mixed
    tool_count: int
    subtask_count: int
    required_roles: tuple[str, ...]
    tags: tuple[str, ...]


def task_to_fingerprint(task: TaskDefinition) -> TaskFingerprint:
    """Convert a TaskDefinition to a TaskFingerprint for pattern matching."""
    return TaskFingerprint(
        task_shape=task.task_shape,
        tool_count=task.tool_count,
        subtask_count=task.subtask_count,
        required_roles=task.required_roles,
        tags=task.tags,
    )


# ---------------------------------------------------------------------------
# Easy sequential tasks (5)
# ---------------------------------------------------------------------------

_EASY_SEQUENTIAL: list[TaskDefinition] = [
    TaskDefinition(
        task_id="easy_seq_01",
        name="Simple Summarization",
        description="Summarize a single document into bullet points.",
        difficulty="easy",
        task_shape="sequential",
        tool_count=1,
        subtask_count=2,
        required_roles=("reader", "writer"),
        tags=("nlp", "summarization"),
    ),
    TaskDefinition(
        task_id="easy_seq_02",
        name="Data Extraction",
        description="Extract structured fields from a form document.",
        difficulty="easy",
        task_shape="sequential",
        tool_count=1,
        subtask_count=2,
        required_roles=("parser", "validator"),
        tags=("extraction", "structured"),
    ),
    TaskDefinition(
        task_id="easy_seq_03",
        name="Translation Pipeline",
        description="Translate text from English to French with quality check.",
        difficulty="easy",
        task_shape="sequential",
        tool_count=1,
        subtask_count=2,
        required_roles=("translator", "reviewer"),
        tags=("nlp", "translation"),
    ),
    TaskDefinition(
        task_id="easy_seq_04",
        name="Code Formatting",
        description="Reformat source code according to style guidelines.",
        difficulty="easy",
        task_shape="sequential",
        tool_count=2,
        subtask_count=2,
        required_roles=("formatter", "linter"),
        tags=("code", "formatting"),
    ),
    TaskDefinition(
        task_id="easy_seq_05",
        name="Email Classification",
        description="Classify incoming emails and route to departments.",
        difficulty="easy",
        task_shape="sequential",
        tool_count=1,
        subtask_count=3,
        required_roles=("classifier", "router", "logger"),
        tags=("classification", "routing"),
    ),
]


# ---------------------------------------------------------------------------
# Medium mixed tasks (8)
# ---------------------------------------------------------------------------

_MEDIUM_MIXED: list[TaskDefinition] = [
    TaskDefinition(
        task_id="med_mix_01",
        name="Research Synthesis",
        description="Search multiple sources, extract findings, synthesize a report.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=3,
        subtask_count=4,
        required_roles=("researcher", "extractor", "synthesizer", "reviewer"),
        tags=("research", "synthesis"),
    ),
    TaskDefinition(
        task_id="med_mix_02",
        name="Code Review Pipeline",
        description="Analyze code for bugs, style, and security issues in parallel, then merge.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=4,
        subtask_count=4,
        required_roles=("bug_finder", "style_checker", "security_auditor", "merger"),
        tags=("code", "review"),
    ),
    TaskDefinition(
        task_id="med_mix_03",
        name="Data Pipeline",
        description="Ingest data, clean it, transform, and load into warehouse.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=3,
        subtask_count=4,
        required_roles=("ingester", "cleaner", "transformer", "loader"),
        tags=("data", "etl"),
    ),
    TaskDefinition(
        task_id="med_mix_04",
        name="Content Creation",
        description="Draft content, generate images, layout, and review.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=4,
        subtask_count=5,
        required_roles=("writer", "artist", "designer", "editor", "reviewer"),
        tags=("content", "creative"),
    ),
    TaskDefinition(
        task_id="med_mix_05",
        name="Incident Triage",
        description="Collect alerts, correlate events, diagnose root cause, propose fix.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=3,
        subtask_count=4,
        required_roles=("collector", "correlator", "diagnostician", "responder"),
        tags=("ops", "incident"),
    ),
    TaskDefinition(
        task_id="med_mix_06",
        name="Document QA",
        description="Chunk document, embed sections, retrieve and answer questions.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=3,
        subtask_count=4,
        required_roles=("chunker", "embedder", "retriever", "answerer"),
        tags=("qa", "rag"),
    ),
    TaskDefinition(
        task_id="med_mix_07",
        name="Test Generation",
        description="Analyze function signatures, generate tests, run them, report coverage.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=3,
        subtask_count=4,
        required_roles=("analyzer", "generator", "runner", "reporter"),
        tags=("testing", "code"),
    ),
    TaskDefinition(
        task_id="med_mix_08",
        name="Multi-Lingual Support",
        description="Detect language, translate query, answer, translate response.",
        difficulty="medium",
        task_shape="mixed",
        tool_count=2,
        subtask_count=4,
        required_roles=("detector", "translator", "responder", "translator_back"),
        tags=("nlp", "multilingual"),
    ),
]


# ---------------------------------------------------------------------------
# Hard parallel tasks (7)
# ---------------------------------------------------------------------------

_HARD_PARALLEL: list[TaskDefinition] = [
    TaskDefinition(
        task_id="hard_par_01",
        name="Full-Stack Feature",
        description="Design API, implement backend, frontend, write tests, deploy -- all in parallel tracks.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=6,
        subtask_count=6,
        required_roles=("architect", "backend_dev", "frontend_dev", "tester", "devops", "reviewer"),
        tags=("engineering", "fullstack"),
    ),
    TaskDefinition(
        task_id="hard_par_02",
        name="Competitive Analysis",
        description="Research 5 competitors in parallel, aggregate findings, produce report.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=5,
        subtask_count=7,
        required_roles=("researcher_1", "researcher_2", "researcher_3", "researcher_4", "researcher_5", "aggregator", "writer"),
        tags=("research", "competitive"),
    ),
    TaskDefinition(
        task_id="hard_par_03",
        name="ML Pipeline",
        description="Feature engineering, model training, hyperparameter tuning, evaluation, deployment.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=5,
        subtask_count=6,
        required_roles=("feature_engineer", "trainer", "tuner", "evaluator", "deployer", "monitor"),
        tags=("ml", "pipeline"),
    ),
    TaskDefinition(
        task_id="hard_par_04",
        name="Security Audit",
        description="SAST, DAST, dependency scan, secrets scan, compliance check -- all parallel.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=5,
        subtask_count=6,
        required_roles=("sast_scanner", "dast_scanner", "dep_scanner", "secret_scanner", "compliance_checker", "reporter"),
        tags=("security", "audit"),
    ),
    TaskDefinition(
        task_id="hard_par_05",
        name="Multi-Modal Processing",
        description="Process text, images, audio, and video streams in parallel, fuse results.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=5,
        subtask_count=5,
        required_roles=("text_processor", "image_processor", "audio_processor", "video_processor", "fusion_engine"),
        tags=("multimodal", "processing"),
    ),
    TaskDefinition(
        task_id="hard_par_06",
        name="Distributed Testing",
        description="Run unit, integration, e2e, perf, and fuzz tests in parallel, aggregate results.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=6,
        subtask_count=6,
        required_roles=("unit_runner", "integration_runner", "e2e_runner", "perf_runner", "fuzz_runner", "aggregator"),
        tags=("testing", "distributed"),
    ),
    TaskDefinition(
        task_id="hard_par_07",
        name="Knowledge Graph Construction",
        description="Extract entities, relations, resolve coreferences, build graph, validate consistency.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=4,
        subtask_count=5,
        required_roles=("entity_extractor", "relation_extractor", "resolver", "graph_builder", "validator"),
        tags=("knowledge", "graph"),
    ),
    TaskDefinition(
        task_id="hard_par_08",
        name="Subtle Bug Detection",
        description="Review production-quality code with non-obvious bugs: off-by-one, race condition, float precision, exception handling.",
        difficulty="hard",
        task_shape="parallel",
        tool_count=4,
        subtask_count=6,
        required_roles=("logic_auditor", "concurrency_auditor", "edge_case_finder", "type_safety_auditor", "integration_reviewer", "reporter"),
        tags=("code", "review", "subtle"),
    ),
]


def get_benchmark_tasks() -> list[TaskDefinition]:
    """Return all 21 benchmark tasks."""
    return list(_EASY_SEQUENTIAL + _MEDIUM_MIXED + _HARD_PARALLEL)
