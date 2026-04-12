"""Tests for CertificateGateComponent — pre-execution integrity check."""

from operon_ai.patterns.certificate_gate import CertificateGateComponent
from operon_ai.patterns.types import InterventionKind, WATCHER_STATE_KEY, WatcherIntervention
from operon_ai.state.dna_repair import DNARepair
from operon_ai.state.genome import Gene, GeneType, Genome


def _make_genome():
    return Genome(
        genes=[
            Gene(name="model", value="test", gene_type=GeneType.STRUCTURAL, required=True),
            Gene(name="temp", value=0.7, gene_type=GeneType.REGULATORY),
        ],
        allow_mutations=True,
        silent=True,
    )


def _mock_stage(name="test_stage"):
    class S:
        pass
    s = S()
    s.name = name
    return s


def test_gate_passes_when_genome_clean():
    """No intervention when genome matches checkpoint."""
    genome = _make_genome()
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    gate = CertificateGateComponent(
        genome=genome, repair=repair, checkpoint=checkpoint,
    )
    gate.on_run_start("task", {})

    shared = {}
    gate.on_stage_start(_mock_stage(), shared, {})

    assert WATCHER_STATE_KEY not in shared
    assert len(gate.blocked_stages) == 0


def test_gate_halts_on_corruption():
    """HALT intervention emitted when genome is corrupted."""
    genome = _make_genome()
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    # Corrupt after checkpoint
    genome.mutate("model", "compromised", reason="attack")

    gate = CertificateGateComponent(
        genome=genome, repair=repair, checkpoint=checkpoint,
    )
    gate.on_run_start("task", {})

    shared = {}
    gate.on_stage_start(_mock_stage("stage_2"), shared, {})

    assert WATCHER_STATE_KEY in shared
    intervention = shared[WATCHER_STATE_KEY]
    assert isinstance(intervention, WatcherIntervention)
    assert intervention.kind == InterventionKind.HALT
    assert "genome corruption" in intervention.reason
    assert "stage_2" == intervention.stage_name
    assert len(gate.blocked_stages) == 1
    assert gate.blocked_stages[0] == "stage_2"


def test_gate_detects_multiple_corruptions():
    """Gate reports all damage types."""
    genome = _make_genome()
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    genome.mutate("model", "bad", reason="attack")
    genome.mutate("temp", 99.9, reason="attack")

    gate = CertificateGateComponent(
        genome=genome, repair=repair, checkpoint=checkpoint,
    )
    gate.on_run_start("task", {})

    shared = {}
    gate.on_stage_start(_mock_stage(), shared, {})

    # Should detect checksum failure + 2 gene drifts = 3 damages
    assert len(gate.damage_reports) >= 2
    assert WATCHER_STATE_KEY in shared


def test_gate_resets_on_run_start():
    """on_run_start clears accumulated state."""
    genome = _make_genome()
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    genome.mutate("model", "bad", reason="attack")

    gate = CertificateGateComponent(
        genome=genome, repair=repair, checkpoint=checkpoint,
    )
    gate.on_run_start("task1", {})

    shared = {}
    gate.on_stage_start(_mock_stage(), shared, {})
    assert len(gate.blocked_stages) == 1

    gate.on_run_start("task2", {})
    assert len(gate.blocked_stages) == 0
    assert len(gate.damage_reports) == 0
