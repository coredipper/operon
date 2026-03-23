"""Tests for adaptive assembly: assemble_pattern, adaptive_skill_organism, AdaptiveSkillOrganism."""

import pytest

from operon_ai import (
    AdaptiveRunResult,
    AdaptiveSkillOrganism,
    PatternLibrary,
    PatternTemplate,
    SkillStage,
    TaskFingerprint,
    adaptive_skill_organism,
)
from operon_ai.patterns.adaptive import assemble_pattern, _auto_fingerprint, _infer_success
from operon_ai.patterns.organism import SkillOrganism
from operon_ai.patterns.review import ReviewerGate
from operon_ai.patterns.swarm import SpecialistSwarm
from operon_ai.patterns.types import SkillRunResult, SkillStageResult
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fp(**kw):
    defaults = dict(task_shape="sequential", tool_count=2, subtask_count=2, required_roles=("worker",))
    defaults.update(kw)
    return TaskFingerprint(**defaults)


def _tmpl(topology="skill_organism", specs=None, fp=None, tid=None):
    lib = PatternLibrary()
    return PatternTemplate(
        template_id=tid or lib.make_id(),
        name=f"test-{topology}",
        topology=topology,
        stage_specs=specs or ({"name": "s1", "role": "Worker", "mode": "fuzzy"},),
        intervention_policy={},
        fingerprint=fp or _fp(),
    )


def _nuclei():
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))
    return fast, deep


def _seeded_library():
    lib = PatternLibrary()
    lib.register_template(_tmpl(
        tid="best",
        topology="skill_organism",
        specs=(
            {"name": "intake", "role": "Normalizer"},
            {"name": "process", "role": "Processor"},
        ),
        fp=_fp(task_shape="sequential", required_roles=("normalizer", "processor")),
    ))
    lib.register_template(_tmpl(tid="other", topology="skill_organism", fp=_fp(task_shape="parallel")))
    return lib


# ---------------------------------------------------------------------------
# assemble_pattern
# ---------------------------------------------------------------------------


def test_assemble_skill_organism_from_template():
    fast, deep = _nuclei()
    tmpl = _tmpl(topology="skill_organism", specs=(
        {"name": "s1", "role": "A"},
        {"name": "s2", "role": "B"},
    ))
    org = assemble_pattern(tmpl, fast_nucleus=fast, deep_nucleus=deep,
                           handlers={"s1": lambda t: "a", "s2": lambda t: "b"})
    assert isinstance(org, SkillOrganism)
    assert len(org.stages) == 2


def test_assemble_single_worker_from_template():
    fast, deep = _nuclei()
    tmpl = _tmpl(topology="single_worker", specs=({"name": "solo", "role": "Solo"},))
    org = assemble_pattern(tmpl, fast_nucleus=fast, deep_nucleus=deep,
                           handlers={"solo": lambda t: "done"})
    assert isinstance(org, SkillOrganism)
    assert len(org.stages) == 1


def test_assemble_reviewer_gate_from_template():
    tmpl = _tmpl(topology="reviewer_gate", specs=(
        {"name": "exec", "role": "Executor"},
        {"name": "rev", "role": "Reviewer"},
    ))
    gate = assemble_pattern(tmpl,
                            handlers={"exec": lambda p: f"done: {p}", "rev": lambda p, c: True})
    assert isinstance(gate, ReviewerGate)


def test_assemble_specialist_swarm_from_template():
    tmpl = _tmpl(topology="specialist_swarm", specs=(
        {"name": "a", "role": "Analyst"},
        {"name": "w", "role": "Writer"},
    ))
    swarm = assemble_pattern(tmpl, workers={
        "Analyst": lambda t: "analysis",
        "Writer": lambda t: "draft",
    })
    assert isinstance(swarm, SpecialistSwarm)


def test_assemble_raises_on_unknown_topology():
    tmpl = _tmpl(topology="quantum_swarm")
    with pytest.raises(ValueError, match="Unknown topology"):
        assemble_pattern(tmpl)


def test_assemble_passes_handlers_to_stages():
    fast, deep = _nuclei()
    handler_fn = lambda task: "handled"
    tmpl = _tmpl(topology="skill_organism", specs=({"name": "h1", "role": "H"},))
    org = assemble_pattern(tmpl, fast_nucleus=fast, deep_nucleus=deep,
                           handlers={"h1": handler_fn})
    assert org.stages[0].handler is handler_fn


# ---------------------------------------------------------------------------
# adaptive_skill_organism
# ---------------------------------------------------------------------------


def test_adaptive_selects_best_template():
    lib = _seeded_library()
    fast, deep = _nuclei()
    adaptive = adaptive_skill_organism(
        "Process data",
        fingerprint=_fp(task_shape="sequential", required_roles=("normalizer",)),
        library=lib,
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"intake": lambda t: "in", "process": lambda t: "out"},
    )
    assert adaptive.template.template_id == "best"


def test_adaptive_auto_fingerprint():
    lib = _seeded_library()
    fast, deep = _nuclei()
    adaptive = adaptive_skill_organism(
        "Do something. Then another thing.",
        library=lib,
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"intake": lambda t: "in", "process": lambda t: "out"},
    )
    assert adaptive.fingerprint.subtask_count == 2


def test_adaptive_raises_on_empty_library():
    fast, deep = _nuclei()
    with pytest.raises(ValueError, match="No templates"):
        adaptive_skill_organism(
            "task",
            library=PatternLibrary(),
            fast_nucleus=fast,
            deep_nucleus=deep,
        )


# ---------------------------------------------------------------------------
# AdaptiveSkillOrganism.run
# ---------------------------------------------------------------------------


def test_adaptive_run_records_outcome():
    lib = _seeded_library()
    fast, deep = _nuclei()
    adaptive = adaptive_skill_organism(
        "Process",
        fingerprint=_fp(),
        library=lib,
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"intake": lambda t: "in", "process": lambda t: "out"},
    )
    result = adaptive.run("Process")
    assert lib.summary()["record_count"] == 1
    assert isinstance(result, AdaptiveRunResult)


def test_adaptive_run_success_heuristic():
    lib = _seeded_library()
    fast, deep = _nuclei()
    adaptive = adaptive_skill_organism(
        "task",
        fingerprint=_fp(),
        library=lib,
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"intake": lambda t: "ok", "process": lambda t: "done"},
    )
    result = adaptive.run("task")
    assert result.record.success is True


def test_adaptive_run_includes_watcher_summary():
    lib = _seeded_library()
    fast, deep = _nuclei()
    adaptive = adaptive_skill_organism(
        "task",
        fingerprint=_fp(),
        library=lib,
        fast_nucleus=fast,
        deep_nucleus=deep,
        handlers={"intake": lambda t: "ok", "process": lambda t: "done"},
    )
    result = adaptive.run("task")
    assert "total_stages_observed" in result.watcher_summary
    assert "convergent" in result.watcher_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_auto_fingerprint_counts_sentences():
    fp = _auto_fingerprint("Do A. Do B. Do C.")
    assert fp.subtask_count == 3
    assert fp.task_shape == "sequential"


def test_infer_success_true_on_execute():
    result = SkillRunResult(
        task="t",
        final_output="ok",
        stage_results=(SkillStageResult(
            stage_name="s", role="r", output="ok", model_alias="fast",
            provider="p", model="m", tokens_used=0, latency_ms=0,
            action_type="EXECUTE", metadata={},
        ),),
        shared_state={},
    )
    assert _infer_success(result) is True


def test_infer_success_false_on_block():
    result = SkillRunResult(
        task="t",
        final_output=None,
        stage_results=(SkillStageResult(
            stage_name="s", role="r", output=None, model_alias="fast",
            provider="p", model="m", tokens_used=0, latency_ms=0,
            action_type="BLOCK", metadata={},
        ),),
        shared_state={},
    )
    assert _infer_success(result) is False
