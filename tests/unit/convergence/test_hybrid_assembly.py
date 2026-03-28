"""Tests for the hybrid assembly module."""

import pytest

from operon_ai import MockProvider, Nucleus
from operon_ai.convergence.hybrid_assembly import (
    _stages_from_template,
    default_template_generator,
    hybrid_skill_organism,
)
from operon_ai.patterns.adaptive import AdaptiveSkillOrganism
from operon_ai.patterns.managed import ManagedOrganism
from operon_ai.patterns.repository import (
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
)
from operon_ai.patterns.types import SkillStage


# ── Helpers ──────────────────────────────────────────────────────


def _make_fingerprint(**overrides) -> TaskFingerprint:
    defaults = dict(
        task_shape="sequential",
        tool_count=0,
        subtask_count=2,
        required_roles=("planner", "executor"),
    )
    defaults.update(overrides)
    return TaskFingerprint(**defaults)


def _make_template(
    template_id: str = "tmpl_001",
    name: str = "good_template",
    fingerprint: TaskFingerprint | None = None,
) -> PatternTemplate:
    fp = fingerprint or _make_fingerprint()
    return PatternTemplate(
        template_id=template_id,
        name=name,
        topology="skill_organism",
        stage_specs=(
            {"name": "planner", "role": "planner", "mode": "fuzzy"},
            {"name": "executor", "role": "executor", "mode": "fuzzy"},
        ),
        intervention_policy={"mode": "default"},
        fingerprint=fp,
        tags=("test",),
    )


def _make_nuclei():
    """Return (fast_nucleus, deep_nucleus) backed by MockProvider."""
    provider = MockProvider(responses={})
    fast = Nucleus(provider=provider)
    deep = Nucleus(provider=provider)
    return fast, deep


def _seed_library_with_high_score(library: PatternLibrary, template: PatternTemplate):
    """Register template and add successful run records so its score is high."""
    library.register_template(template)
    for i in range(5):
        library.record_run(
            PatternRunRecord(
                record_id=f"run_{i}",
                template_id=template.template_id,
                fingerprint=template.fingerprint,
                success=True,
                latency_ms=100.0,
                tokens_used=50,
            )
        )


# ── Tests ────────────────────────────────────────────────────────


class TestHybridUsesLibraryWhenScoreAboveThreshold:
    def test_returns_adaptive_organism(self):
        """When the library has a high-scoring template, the adaptive path is taken."""
        library = PatternLibrary()
        fp = _make_fingerprint()
        template = _make_template(fingerprint=fp)
        _seed_library_with_high_score(library, template)
        fast, deep = _make_nuclei()

        result = hybrid_skill_organism(
            "Summarize this document.",
            library=library,
            fingerprint=fp,
            fast_nucleus=fast,
            deep_nucleus=deep,
            score_threshold=0.5,
        )

        assert isinstance(result, AdaptiveSkillOrganism)


class TestHybridFallsBackToGenerator:
    def test_uses_generator_with_empty_library(self):
        """Empty library + generator provided -> managed organism from generated template."""
        library = PatternLibrary()
        fp = _make_fingerprint()
        fast, deep = _make_nuclei()

        generated_templates: list[PatternTemplate] = []

        def tracking_generator(task: str) -> PatternTemplate:
            t = default_template_generator(task)
            generated_templates.append(t)
            return t

        result = hybrid_skill_organism(
            "Build a report.",
            library=library,
            fingerprint=fp,
            fast_nucleus=fast,
            deep_nucleus=deep,
            template_generator=tracking_generator,
            score_threshold=0.5,
        )

        assert isinstance(result, ManagedOrganism)
        assert len(generated_templates) == 1


class TestHybridRaisesWithoutGeneratorOrTemplates:
    def test_raises_value_error(self):
        """Empty library and no generator -> ValueError."""
        library = PatternLibrary()
        fp = _make_fingerprint()
        fast, deep = _make_nuclei()

        with pytest.raises(ValueError, match="No templates available and no generator provided"):
            hybrid_skill_organism(
                "Do something.",
                library=library,
                fingerprint=fp,
                fast_nucleus=fast,
                deep_nucleus=deep,
            )


class TestDefaultTemplateGenerator:
    def test_produces_valid_template(self):
        """default_template_generator returns a PatternTemplate with 3 stages."""
        template = default_template_generator("Plan the sprint. Execute tasks. Review output.")

        assert isinstance(template, PatternTemplate)
        assert template.topology == "skill_organism"
        assert len(template.stage_specs) == 3

        names = [s["name"] for s in template.stage_specs]
        assert names == ["planner", "executor", "reviewer"]

        modes = [s["mode"] for s in template.stage_specs]
        assert modes == ["fuzzy", "fuzzy", "fixed"]

    def test_fingerprint_derived_from_task(self):
        """Fingerprint subtask_count reflects number of sentences."""
        template = default_template_generator("Step one. Step two. Step three.")
        assert template.fingerprint.subtask_count == 3
        assert template.fingerprint.task_shape == "sequential"
        assert template.fingerprint.required_roles == ("planner", "executor", "reviewer")


class TestStagesFromTemplate:
    def test_converts_to_skill_stages(self):
        """_stages_from_template converts stage_specs into SkillStage instances."""
        template = _make_template()
        stages = _stages_from_template(template)

        assert len(stages) == 2
        assert all(isinstance(s, SkillStage) for s in stages)

        assert stages[0].name == "planner"
        assert stages[0].role == "planner"
        assert stages[0].mode == "fuzzy"

        assert stages[1].name == "executor"
        assert stages[1].role == "executor"
        assert stages[1].mode == "fuzzy"

    def test_placeholder_handlers_are_callable(self):
        """Each generated stage has a callable placeholder handler."""
        template = _make_template()
        stages = _stages_from_template(template)

        for stage in stages:
            assert stage.handler is not None
            result = stage.handler("test task", {}, {}, stage)
            assert isinstance(result, str)
            assert stage.name in result

    def test_three_stage_template(self):
        """Works with the 3-stage template from default_template_generator."""
        template = default_template_generator("Do something.")
        stages = _stages_from_template(template)

        assert len(stages) == 3
        assert stages[0].name == "planner"
        assert stages[1].name == "executor"
        assert stages[2].name == "reviewer"
        assert stages[2].mode == "fixed"


class TestGeneratorTemplateRegisteredInLibrary:
    def test_template_registered_after_fallback(self):
        """After fallback to generator, the library contains the generated template."""
        library = PatternLibrary()
        fp = _make_fingerprint()
        fast, deep = _make_nuclei()

        assert library.summary()["template_count"] == 0

        hybrid_skill_organism(
            "Analyze data trends.",
            library=library,
            fingerprint=fp,
            fast_nucleus=fast,
            deep_nucleus=deep,
            template_generator=default_template_generator,
            score_threshold=0.5,
        )

        # Library should now contain at least the generated template
        assert library.summary()["template_count"] >= 1

        # Verify it is a skill_organism template with the expected stages
        templates = library.retrieve_templates(topology="skill_organism")
        assert len(templates) >= 1
        generated = [t for t in templates if "generated" in t.tags]
        assert len(generated) == 1
        assert len(generated[0].stage_specs) == 3
