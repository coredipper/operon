"""Tests for the LangGraph compiler wrapper.

Uses handler-backed stages (no LLM needed). Skipped if LangGraph
is not installed.
"""

from __future__ import annotations

import pytest

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.langgraph_compiler import (
    HAS_LANGGRAPH,
    run_organism_langgraph,
)

pytestmark = pytest.mark.skipif(
    not HAS_LANGGRAPH, reason="LangGraph not installed"
)


def _make_organism(stages=None, components=None, budget=1000):
    """Build an organism with handler-backed stages (no LLM)."""
    if stages is None:
        stages = [
            SkillStage(name="reader", role="Reader", handler=lambda task: f"Read: {task}"),
            SkillStage(name="writer", role="Writer", handler=lambda task: f"Wrote: {task}"),
        ]
    return skill_organism(
        stages=stages,
        fast_nucleus=Nucleus(provider=MockProvider()),
        deep_nucleus=Nucleus(provider=MockProvider()),
        budget=ATP_Store(budget=budget),
        components=list(components or []),
    )


class TestNormalExecution:
    def test_two_stage_pipeline(self):
        org = _make_organism()
        r = run_organism_langgraph(org, task="hello")
        assert r.metadata["stages_completed"] == ["reader", "writer"]
        assert r.metadata["halted"] is False
        assert r.output != ""

    def test_single_stage(self):
        org = _make_organism(stages=[
            SkillStage(name="s", role="R", handler=lambda task: f"done: {task}"),
        ])
        r = run_organism_langgraph(org, task="test")
        assert r.metadata["stages_completed"] == ["s"]
        assert "done: test" in r.output

    def test_certificates_verified(self):
        org = _make_organism()
        r = run_organism_langgraph(org, task="test")
        assert len(r.certificates_verified) >= 1
        assert all(c["holds"] for c in r.certificates_verified)


class TestCertificateGateHalt:
    def test_halts_on_genome_corruption(self):
        from operon_ai.patterns.certificate_gate import CertificateGateComponent
        from operon_ai.state.genome import Genome, Gene, GeneType
        from operon_ai.state.dna_repair import DNARepair

        genome = Genome(
            genes=[Gene(name="m", gene_type=GeneType.REGULATORY, value="ok")],
            allow_mutations=True,
        )
        repair = DNARepair(silent=True)
        cp = repair.checkpoint(genome)
        genome.mutate("m", "BAD")
        gate = CertificateGateComponent(genome=genome, repair=repair, checkpoint=cp)

        org = _make_organism(
            stages=[SkillStage(name="s", role="R", handler=lambda task: "done")],
            components=[gate],
        )
        r = run_organism_langgraph(org, task="test")
        assert r.metadata["halted"] is True
        assert r.metadata["stages_completed"] == []


class TestWatcherHalt:
    def test_halt_on_final_stage(self):
        """Watcher HALT on the final stage must set halted=True."""
        from operon_ai.patterns.watcher import WatcherComponent, WatcherConfig

        # Watcher with very low intervention rate → halts quickly
        watcher = WatcherComponent(
            config=WatcherConfig(max_intervention_rate=0.0),
        )

        # Stage that produces a FAILURE result to trigger watcher
        def failing_handler(task):
            raise ValueError("intentional failure")

        org = _make_organism(
            stages=[
                SkillStage(name="s1", role="R", handler=lambda task: "ok"),
                SkillStage(name="s2", role="W", handler=failing_handler),
            ],
            components=[watcher],
        )
        r = run_organism_langgraph(org, task="test")
        # Should be halted — either by halt_on_block or watcher
        assert r.metadata["halted"] is True


class TestExceptionHandling:
    def test_exception_returns_error_result(self):
        """Exceptions in organism.run() produce a stable error result."""
        org = _make_organism(stages=[
            SkillStage(name="s", role="R", handler=lambda task: (_ for _ in ()).throw(RuntimeError("boom"))),
        ])
        r = run_organism_langgraph(org, task="test")
        assert r.metadata["halted"] is True
        assert "Error" in r.output or "boom" in r.output
