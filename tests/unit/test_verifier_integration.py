"""Integration tests: VerifierComponent + WatcherComponent + SkillOrganism."""

import pytest

from operon_ai import ATP_Store, Nucleus, SkillStage, skill_organism
from operon_ai.patterns.verifier import VerifierComponent, VerifierConfig
from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig
from operon_ai.patterns.types import InterventionKind
from operon_ai.providers.mock import MockProvider


def _make_nuclei():
    provider = MockProvider()
    fast = Nucleus(provider=provider, base_energy_cost=10)
    deep = Nucleus(provider=provider, base_energy_cost=30)
    return fast, deep


def test_verifier_triggers_escalation_on_low_quality():
    """Low quality on fast model triggers ESCALATE via verifier signal.

    Verifies that component ordering doesn't matter: verifier deposits
    signal, watcher picks it up regardless of component list order.
    """
    fast, deep = _make_nuclei()
    budget = ATP_Store(budget=500, silent=True)

    # Rubric always returns low quality
    def rubric(output, stage_name):
        return 0.2

    watcher = WatcherComponent(config=WatcherConfig(), budget=budget)
    verifier = VerifierComponent(
        rubric=rubric,
        config=VerifierConfig(quality_low_threshold=0.5),
    )

    # Documented order: watcher before verifier — should still work
    organism = skill_organism(
        stages=[
            SkillStage(name="review", role="reviewer",
                       instructions="Review this.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[watcher, verifier],
        budget=budget,
    )

    result = organism.run("test task")

    # Watcher should have seen the verifier signal and escalated
    escalated = any(
        i.kind == InterventionKind.ESCALATE for i in watcher.interventions
    )
    assert escalated, (
        f"Expected ESCALATE from low quality, got interventions: "
        f"{[(i.kind.value, i.reason) for i in watcher.interventions]}"
    )


def test_verifier_no_escalation_on_high_quality():
    """High quality does not trigger escalation."""
    fast, deep = _make_nuclei()
    budget = ATP_Store(budget=500, silent=True)

    def rubric(output, stage_name):
        return 0.9

    watcher = WatcherComponent(config=WatcherConfig(), budget=budget)
    verifier = VerifierComponent(rubric=rubric)

    organism = skill_organism(
        stages=[
            SkillStage(name="review", role="reviewer",
                       instructions="Review this.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[watcher, verifier],
        budget=budget,
    )

    result = organism.run("test task")

    escalated = any(
        i.kind == InterventionKind.ESCALATE for i in watcher.interventions
    )
    assert not escalated


def test_certificate_gate_blocks_before_execution():
    """CertificateGate HALT preserves intervention info in shared_state."""
    from operon_ai.patterns.certificate_gate import CertificateGateComponent
    from operon_ai.state.dna_repair import DNARepair
    from operon_ai.state.genome import Gene, GeneType, Genome

    fast, deep = _make_nuclei()
    budget = ATP_Store(budget=500, silent=True)

    genome = Genome(
        genes=[Gene(name="model", value="test", gene_type=GeneType.STRUCTURAL, required=True)],
        allow_mutations=True,
        silent=True,
    )
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    # Corrupt before running
    genome.mutate("model", "compromised", reason="attack")

    gate = CertificateGateComponent(
        genome=genome, repair=repair, checkpoint=checkpoint,
    )

    organism = skill_organism(
        stages=[
            SkillStage(name="stage1", role="worker",
                       instructions="Do work.", mode="fast"),
            SkillStage(name="stage2", role="worker",
                       instructions="More work.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[gate],
        budget=budget,
    )

    result = organism.run("test task")

    # Should have blocked before stage1 executed
    assert result.final_output is None
    assert "_blocked_by" in result.shared_state
    blocked = result.shared_state["_blocked_by"]
    assert blocked.kind == InterventionKind.HALT
    assert "genome corruption" in blocked.reason
