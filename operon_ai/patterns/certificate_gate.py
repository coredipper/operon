"""Certificate gate — pre-execution integrity verification.

Biological Analogy: G1/S DNA Damage Checkpoint
- In cell biology, the G1/S checkpoint halts cell division if DNA damage
  is detected BEFORE replication begins.  This prevents corrupted DNA
  from propagating to daughter cells.
- CertificateGateComponent checks genome integrity BEFORE each stage
  executes its LLM call.  If corruption is detected, it emits a HALT
  intervention, preventing the corrupted state from affecting execution.

This moves DNARepair from reactive (detect-after-corrupt) to preventive
(block-before-execute), completing the cell cycle analogy alongside the
existing CellCycleController.

References:
  Rashie & Rashi (arXiv:2604.01483) — pre-execution verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import (
    InterventionKind,
    SkillRunResult,
    SkillStage,
    SkillStageResult,
    WatcherIntervention,
    WATCHER_STATE_KEY,
)

if TYPE_CHECKING:
    from ..state.dna_repair import DNARepair, StateCheckpoint
    from ..state.genome import Genome


@dataclass
class CertificateGateComponent:
    """Pre-execution integrity check that halts on genome corruption.

    Scans the genome against a checkpoint before each stage executes.
    If damage is detected, writes a HALT intervention to shared_state
    so the organism stops before the corrupted state reaches the LLM.

    Usage::

        genome = Genome(...)
        repair = DNARepair(silent=True)
        checkpoint = repair.checkpoint(genome)

        gate = CertificateGateComponent(
            genome=genome,
            repair=repair,
            checkpoint=checkpoint,
        )
        organism = skill_organism(..., components=[watcher, gate])
    """

    genome: Genome
    repair: DNARepair
    checkpoint: StateCheckpoint

    # Per-run state
    blocked_stages: list[str] = field(default_factory=list)
    damage_reports: list[Any] = field(default_factory=list)

    # -- SkillRuntimeComponent protocol ----------------------------------

    def on_run_start(self, task: str, shared_state: dict[str, Any]) -> None:
        """Reset per-run state."""
        self.blocked_stages.clear()
        self.damage_reports.clear()

    def on_stage_start(
        self,
        stage: SkillStage,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """Pre-flight integrity check: scan genome before stage executes."""
        damage = self.repair.scan(self.genome, self.checkpoint)
        if damage:
            stage_name = getattr(stage, "name", "unknown")
            self.blocked_stages.append(stage_name)
            self.damage_reports.extend(damage)

            # Emit HALT intervention via shared_state.
            # Use the RunContext's dynamic watcher key when available,
            # falling back to the default WATCHER_STATE_KEY.
            from .types import RunContext
            key = shared_state._watcher_key if isinstance(shared_state, RunContext) else WATCHER_STATE_KEY
            shared_state[key] = WatcherIntervention(
                kind=InterventionKind.HALT,
                stage_name=stage_name,
                reason=f"genome corruption detected before stage: {len(damage)} damage(s)",
            )

    def on_stage_result(
        self,
        stage: SkillStage,
        result: SkillStageResult,
        shared_state: dict[str, Any],
        stage_outputs: dict[str, Any],
    ) -> None:
        """No post-stage action needed."""

    def on_run_complete(
        self,
        result: SkillRunResult,
        shared_state: dict[str, Any],
    ) -> None:
        """No post-run action needed."""
