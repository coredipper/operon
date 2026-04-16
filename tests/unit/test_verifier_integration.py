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


# ---------------------------------------------------------------------------
# Behavioral certificate integration tests
# ---------------------------------------------------------------------------


def test_verifier_certify_behavior():
    """VerifierComponent produces a behavioral_quality certificate after a run."""
    fast, deep = _make_nuclei()
    budget = ATP_Store(budget=500, silent=True)

    def rubric(output, stage_name):
        return 0.9

    verifier = VerifierComponent(rubric=rubric)
    organism = skill_organism(
        stages=[
            SkillStage(name="s1", role="worker", instructions="Do work.", mode="fast"),
            SkillStage(name="s2", role="worker", instructions="More.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[verifier],
        budget=budget,
    )
    organism.run("test task")

    cert = verifier.certify_behavior(threshold=0.8)
    assert cert is not None
    assert cert.theorem == "behavioral_quality"
    result = cert.verify()
    assert result.holds is True
    assert result.evidence["n"] == 2


def test_verifier_certify_behavior_fails_low_quality():
    """behavioral_quality cert fails when rubric scores are below threshold."""
    fast, deep = _make_nuclei()
    budget = ATP_Store(budget=500, silent=True)

    def rubric(output, stage_name):
        return 0.3

    verifier = VerifierComponent(rubric=rubric)
    organism = skill_organism(
        stages=[
            SkillStage(name="s1", role="worker", instructions="Do work.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[verifier],
        budget=budget,
    )
    organism.run("test task")

    cert = verifier.certify_behavior(threshold=0.8)
    assert cert is not None
    result = cert.verify()
    assert result.holds is False


def test_verifier_certify_behavior_none_without_rubric():
    """No certificate produced when no rubric was configured."""
    verifier = VerifierComponent(rubric=None)
    assert verifier.certify_behavior() is None


def test_watcher_certify_behavior_stability_healthy():
    """Stability cert holds when epiplexity severity is low (healthy)."""
    from operon_ai.patterns.watcher import SignalCategory, WatcherSignal

    watcher = WatcherComponent()
    # Signal values are already severity (high = worse) after normalization
    watcher.signals = [
        WatcherSignal(category=SignalCategory.EPISTEMIC, source="epiplexity",
                      stage_name="s1", value=0.2),
        WatcherSignal(category=SignalCategory.EPISTEMIC, source="epiplexity",
                      stage_name="s2", value=0.3),
        # verifier signal — should be excluded from stability cert
        WatcherSignal(category=SignalCategory.EPISTEMIC, source="verifier",
                      stage_name="s1", value=0.8),
    ]
    certs = watcher.certify_behavior(threshold=0.5)
    stability = [c for c in certs if c.theorem == "behavioral_stability"]
    assert len(stability) == 1
    result = stability[0].verify()
    assert result.holds is True
    # Only epiplexity signals (0.2, 0.3), not verifier (0.8): mean = 0.25
    assert result.evidence["mean"] == 0.25
    assert result.evidence["n"] == 2


def test_watcher_certify_behavior_stability_stagnant():
    """Stability cert fails when epiplexity severity is high (stagnant)."""
    from operon_ai.patterns.watcher import SignalCategory, WatcherSignal

    watcher = WatcherComponent()
    watcher.signals = [
        WatcherSignal(category=SignalCategory.EPISTEMIC, source="epiplexity",
                      stage_name="s1", value=0.9),
        WatcherSignal(category=SignalCategory.EPISTEMIC, source="epiplexity",
                      stage_name="s2", value=0.85),
    ]
    certs = watcher.certify_behavior(threshold=0.5)
    stability = [c for c in certs if c.theorem == "behavioral_stability"]
    assert len(stability) == 1
    result = stability[0].verify()
    assert result.holds is False  # stagnant run should NOT get stability cert


def test_watcher_certify_behavior_no_anomaly():
    """WatcherComponent produces behavioral_no_anomaly from immune signals."""
    from operon_ai.patterns.watcher import SignalCategory, WatcherSignal

    watcher = WatcherComponent()
    watcher.signals = [
        WatcherSignal(category=SignalCategory.SPECIES_SPECIFIC, source="immune_system",
                      stage_name="s1", value=0.0,
                      detail={"threat_level": "none", "action": "ignore"}),
        WatcherSignal(category=SignalCategory.SPECIES_SPECIFIC, source="immune_system",
                      stage_name="s2", value=0.3,
                      detail={"threat_level": "suspicious", "action": "monitor"}),
    ]
    certs = watcher.certify_behavior()
    anomaly = [c for c in certs if c.theorem == "behavioral_no_anomaly"]
    assert len(anomaly) == 1
    result = anomaly[0].verify()
    assert result.holds is True
    assert result.evidence["max_threat"] == "suspicious"


def test_watcher_certify_behavior_empty_without_signals():
    """Empty list when no relevant signals exist."""
    watcher = WatcherComponent()
    assert watcher.certify_behavior() == []


def test_collect_certificates_includes_behavioral():
    """SkillOrganism.collect_certificates() picks up behavioral certs."""
    fast, deep = _make_nuclei()
    budget = ATP_Store(budget=500, silent=True)

    def rubric(output, stage_name):
        return 0.85

    verifier = VerifierComponent(rubric=rubric)
    watcher = WatcherComponent(budget=budget)

    organism = skill_organism(
        stages=[
            SkillStage(name="s1", role="worker", instructions="Work.", mode="fast"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        components=[watcher, verifier],
        budget=budget,
    )
    organism.run("test task")

    certs = organism.collect_certificates()
    theorems = [c.theorem for c in certs]
    assert "priority_gating" in theorems  # structural (from ATP_Store)
    assert "behavioral_quality" in theorems  # from verifier
    # watcher behavioral certs are lists — should be flattened in
    for cert in certs:
        assert cert.verify().holds is True
