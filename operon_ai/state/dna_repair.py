"""
DNA Repair: State Integrity Checking and Recovery
==================================================

Biological Analogy:
- Mismatch Repair (MMR): Detects and fixes replication errors
  → Expression drift: gene expression changed without regulatory signal
- Base Excision Repair (BER): Removes damaged bases one at a time
  → Genome drift: runtime config diverging from checkpointed state
- Nucleotide Excision Repair (NER): Removes bulky DNA lesions
  → Memory corruption: BiTemporalMemory contradictions
- Double-Strand Break Repair: Homologous recombination or NHEJ
  → Checksum failure: genome hash mismatch, most severe corruption

The DNARepair system fills the gap between surveillance (external threats
via InnateImmunity) and healing (output/agent recovery via ChaperoneLoop
and RegenerativeSwarm).  DNA repair is *proactive*: it checks internal
state integrity before corruption propagates downstream.

Key difference from existing healing:
- ChaperoneLoop fixes output structure (protein folding)
- RegenerativeSwarm fixes stuck agents (apoptosis + regeneration)
- DNARepair fixes internal STATE (genome integrity)

Integration points:
- Genome: checksumming, expression validation, mutation rollback
- HistoneStore: successful repairs stored as epigenetic markers
- Certificate: issues ``state_integrity_verified`` certificates
- BiTemporalMemory: detects fact contradictions and supersession loops

Example::

    genome = Genome(genes=[Gene("model", "gpt-4", required=True)], silent=True)
    repair = DNARepair(silent=True)
    checkpoint = repair.checkpoint(genome)

    genome.mutate("model", "gpt-3.5", reason="cost reduction")
    damage = repair.scan(genome, checkpoint)
    # [DamageReport(corruption_type=GENOME_DRIFT, ...)]

    for d in damage:
        repair.repair(genome, d)

    cert = repair.certify(genome, checkpoint)
    assert cert.verify().holds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, TYPE_CHECKING
import uuid

from ..core.certificate import Certificate, register_verify_fn as _register

if TYPE_CHECKING:
    from .genome import Genome
    from .histone import HistoneStore
    from ..memory.bitemporal import BiTemporalMemory


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CorruptionType(Enum):
    """Types of state corruption (analogous to DNA lesion types)."""
    GENOME_DRIFT = "genome_drift"           # BER analog: gene values changed
    EXPRESSION_DRIFT = "expression_drift"   # MMR analog: expression without regulatory signal
    MEMORY_CORRUPTION = "memory_corruption" # NER analog: contradictions in BiTemporalMemory
    CHECKSUM_FAILURE = "checksum_failure"    # DSB analog: hash mismatch


class DamageSeverity(IntEnum):
    """Severity levels for detected damage."""
    LOW = 1        # Single gene drift, minor expression change
    MODERATE = 2   # Multiple gene drifts, memory inconsistency
    HIGH = 3       # Checksum failure, widespread corruption
    CRITICAL = 4   # Irrecoverable without full checkpoint restore


class RepairStrategy(Enum):
    """Repair approaches (analogous to DNA repair pathways)."""
    ROLLBACK = "rollback"                     # Revert to checkpoint gene values
    RE_EXPRESS = "re_express"                 # Reset expression levels to defaults
    EPIGENETIC_PATCH = "epigenetic_patch"     # Store repair as histone marker
    CHECKPOINT_RESTORE = "checkpoint_restore" # Full state restore from checkpoint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateCheckpoint:
    """Snapshot of genome state for later comparison.

    Biological analog: DNA checkpoint before cell division — a reference
    state that the repair machinery compares against.
    """
    genome_hash: str
    expression_snapshot: tuple[tuple[str, int], ...]  # (gene_name, level_value) pairs
    gene_values: tuple[tuple[str, Any], ...]  # (gene_name, value) pairs
    gene_count: int
    timestamp: datetime
    checkpoint_id: str

    @property
    def expression_dict(self) -> dict[str, int]:
        """Expression snapshot as a dict for convenience."""
        return dict(self.expression_snapshot)

    @property
    def values_dict(self) -> dict[str, Any]:
        """Gene values as a dict for convenience."""
        return dict(self.gene_values)


@dataclass(frozen=True)
class DamageReport:
    """A single detected corruption.

    Biological analog: The lesion site identified by a repair enzyme —
    contains the type of damage, its location, and recommended fix.
    """
    corruption_type: CorruptionType
    severity: DamageSeverity
    location: str            # e.g. "gene:temperature" or "expression:model"
    description: str         # Human-readable explanation
    expected: Any
    actual: Any
    recommended_strategy: RepairStrategy


@dataclass
class RepairResult:
    """Outcome of a repair operation."""
    success: bool
    strategy_used: RepairStrategy
    damage: DamageReport
    details: str
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Verification function for Certificate integration
# ---------------------------------------------------------------------------

def _verify_state_integrity(
    params: Any,
) -> tuple[bool, dict[str, Any]]:
    """Verify that genome state matches checkpoint.

    Checks genome_hash equality and gene_count match.
    """
    genome_hash = params["genome_hash"]
    checkpoint_hash = params["checkpoint_hash"]
    gene_count = params["gene_count"]
    checkpoint_gene_count = params["checkpoint_gene_count"]

    hash_match = genome_hash == checkpoint_hash
    count_match = gene_count == checkpoint_gene_count
    holds = hash_match and count_match

    return holds, {
        "genome_hash": genome_hash,
        "checkpoint_hash": checkpoint_hash,
        "hash_match": hash_match,
        "gene_count": gene_count,
        "checkpoint_gene_count": checkpoint_gene_count,
        "count_match": count_match,
    }


_register("state_integrity_verified", _verify_state_integrity)


# ---------------------------------------------------------------------------
# DNARepair engine
# ---------------------------------------------------------------------------

class DNARepair:
    """State integrity checker and repair engine.

    Biological analog: The DNA damage response (DDR) system — a
    coordinated set of enzymes that detect, signal, and repair
    damage to the genome.

    Args:
        histone_store: Optional store for persisting repair lessons
            as epigenetic markers.
        auto_repair: If True, :meth:`scan` automatically applies
            recommended repairs for all detected damage.
        silent: Suppress console output.
    """

    def __init__(
        self,
        histone_store: HistoneStore | None = None,
        auto_repair: bool = False,
        silent: bool = False,
    ):
        self._histone_store = histone_store
        self._auto_repair = auto_repair
        self.silent = silent

        self._checkpoints: dict[str, StateCheckpoint] = {}
        self._repair_history: list[RepairResult] = []
        self._scan_count: int = 0

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def checkpoint(self, genome: Genome) -> StateCheckpoint:
        """Create a snapshot of genome state for later comparison.

        Args:
            genome: The genome to snapshot.

        Returns:
            A frozen :class:`StateCheckpoint`.
        """
        expression_pairs = tuple(
            (name, state.level.value)
            for name, state in sorted(genome._expression.items())
        )
        gene_value_pairs = tuple(
            (name, gene.value)
            for name, gene in sorted(genome._genes.items())
        )

        cp = StateCheckpoint(
            genome_hash=genome.get_hash(),
            expression_snapshot=expression_pairs,
            gene_values=gene_value_pairs,
            gene_count=len(genome._genes),
            timestamp=datetime.now(),
            checkpoint_id=str(uuid.uuid4())[:8],
        )
        self._checkpoints[cp.checkpoint_id] = cp

        if not self.silent:
            print(
                f"\U0001f9ec [DNARepair] Checkpoint {cp.checkpoint_id}: "
                f"hash={cp.genome_hash}, genes={cp.gene_count}"
            )
        return cp

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def scan(
        self,
        genome: Genome,
        checkpoint: StateCheckpoint,
    ) -> list[DamageReport]:
        """Scan genome state against a checkpoint for corruption.

        Performs four independent scans:

        1. **Checksum** — genome hash mismatch (DSB analog)
        2. **Gene count** — genes added or removed (genome drift)
        3. **Expression** — expression levels changed without a
           corresponding approved mutation (MMR analog)
        4. **Required genes** — required genes that are silenced

        Returns:
            Damage reports sorted by severity (highest first).
        """
        from .genome import ExpressionLevel

        self._scan_count += 1
        damage: list[DamageReport] = []

        # 1. Checksum scan (DSB)
        current_hash = genome.get_hash()
        if current_hash != checkpoint.genome_hash:
            damage.append(DamageReport(
                corruption_type=CorruptionType.CHECKSUM_FAILURE,
                severity=DamageSeverity.HIGH,
                location="genome:hash",
                description=(
                    f"Genome hash changed: "
                    f"{checkpoint.genome_hash} → {current_hash}"
                ),
                expected=checkpoint.genome_hash,
                actual=current_hash,
                recommended_strategy=RepairStrategy.ROLLBACK,
            ))

        # 2. Individual gene value scan (BER analog)
        cp_values = checkpoint.values_dict
        for gene_name, expected_value in cp_values.items():
            gene = genome.get_gene(gene_name)
            if gene is None:
                damage.append(DamageReport(
                    corruption_type=CorruptionType.GENOME_DRIFT,
                    severity=DamageSeverity.MODERATE,
                    location=f"gene:{gene_name}",
                    description=f"Gene '{gene_name}' was removed",
                    expected=expected_value,
                    actual=None,
                    recommended_strategy=RepairStrategy.ROLLBACK,
                ))
            elif gene.value != expected_value:
                damage.append(DamageReport(
                    corruption_type=CorruptionType.GENOME_DRIFT,
                    severity=DamageSeverity.MODERATE,
                    location=f"gene:{gene_name}",
                    description=(
                        f"Gene '{gene_name}' value changed: "
                        f"{expected_value} → {gene.value}"
                    ),
                    expected=expected_value,
                    actual=gene.value,
                    recommended_strategy=RepairStrategy.ROLLBACK,
                ))

        # 3. Gene count scan
        current_count = len(genome._genes)
        if current_count != checkpoint.gene_count:
            damage.append(DamageReport(
                corruption_type=CorruptionType.GENOME_DRIFT,
                severity=DamageSeverity.MODERATE,
                location="genome:gene_count",
                description=(
                    f"Gene count changed: "
                    f"{checkpoint.gene_count} → {current_count}"
                ),
                expected=checkpoint.gene_count,
                actual=current_count,
                recommended_strategy=RepairStrategy.CHECKPOINT_RESTORE,
            ))

        # 4. Expression scan (MMR)
        cp_expr = checkpoint.expression_dict
        for gene_name, expected_level in cp_expr.items():
            if gene_name not in genome._expression:
                continue
            actual_level = genome._expression[gene_name].level.value
            if actual_level != expected_level:
                # Check if this is a legitimately regulated change
                modifier = genome._expression[gene_name].modifier
                if modifier and modifier != "":
                    # Has a modifier — could be legitimate, but still report
                    # with lower severity
                    severity = DamageSeverity.LOW
                else:
                    severity = DamageSeverity.MODERATE

                damage.append(DamageReport(
                    corruption_type=CorruptionType.EXPRESSION_DRIFT,
                    severity=severity,
                    location=f"expression:{gene_name}",
                    description=(
                        f"Expression of '{gene_name}' changed: "
                        f"{ExpressionLevel(expected_level).name} → "
                        f"{ExpressionLevel(actual_level).name}"
                    ),
                    expected=expected_level,
                    actual=actual_level,
                    recommended_strategy=RepairStrategy.RE_EXPRESS,
                ))

        # 5. Required gene validation
        valid, errors = genome.validate()
        if not valid:
            for error in errors:
                damage.append(DamageReport(
                    corruption_type=CorruptionType.EXPRESSION_DRIFT,
                    severity=DamageSeverity.HIGH,
                    location=f"validation:{error}",
                    description=error,
                    expected="expressed",
                    actual="silenced",
                    recommended_strategy=RepairStrategy.RE_EXPRESS,
                ))

        # Sort by severity descending
        damage.sort(key=lambda d: d.severity.value, reverse=True)

        if not self.silent:
            if damage:
                print(
                    f"\U0001f9ec [DNARepair] Scan #{self._scan_count}: "
                    f"{len(damage)} damage site(s) detected"
                )
                for d in damage:
                    print(
                        f"  \u26a0\ufe0f {d.corruption_type.value} "
                        f"[{d.severity.name}] at {d.location}"
                    )
            else:
                print(
                    f"\U0001f9ec [DNARepair] Scan #{self._scan_count}: "
                    f"no damage detected"
                )

        # Auto-repair if enabled
        if self._auto_repair and damage:
            for d in damage:
                self.repair(genome, d)

        return damage

    def scan_memory(
        self,
        memory: BiTemporalMemory,
    ) -> list[DamageReport]:
        """Scan BiTemporalMemory for corruption.

        Detects:
        - Circular supersession chains (fact A supersedes B supersedes A)

        Args:
            memory: The bi-temporal memory to scan.

        Returns:
            Damage reports for any detected corruption.
        """
        self._scan_count += 1
        damage: list[DamageReport] = []

        # Build supersession graph: supersedes → fact_id
        # BiTemporalFact uses `supersedes` (the ID of the fact this one replaces)
        supersession: dict[str, str] = {}  # old_id → new_id
        for fact in memory._facts:
            if fact.supersedes is not None:
                supersession[fact.supersedes] = fact.fact_id

        # Check for circular supersession chains
        seen: set[str] = set()
        for start_id in supersession:
            if start_id in seen:
                continue
            chain: list[str] = [start_id]
            current_id = supersession.get(start_id)
            while current_id and current_id not in seen:
                if current_id in chain:
                    damage.append(DamageReport(
                        corruption_type=CorruptionType.MEMORY_CORRUPTION,
                        severity=DamageSeverity.HIGH,
                        location=f"memory:supersession:{start_id}",
                        description=(
                            f"Circular supersession chain detected: "
                            f"{' → '.join(chain)} → {current_id}"
                        ),
                        expected="linear supersession",
                        actual="circular chain",
                        recommended_strategy=RepairStrategy.EPIGENETIC_PATCH,
                    ))
                    break
                chain.append(current_id)
                current_id = supersession.get(current_id)
            seen.update(chain)

        if not self.silent:
            if damage:
                print(
                    f"\U0001f9ec [DNARepair] Memory scan: "
                    f"{len(damage)} corruption(s) detected"
                )
            else:
                print(
                    f"\U0001f9ec [DNARepair] Memory scan: "
                    f"no corruption detected"
                )

        return damage

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair(
        self,
        genome: Genome,
        damage: DamageReport,
        strategy: RepairStrategy | None = None,
    ) -> RepairResult:
        """Apply a repair to the genome.

        Args:
            genome: The genome to repair.
            damage: The damage to fix.
            strategy: Override the recommended strategy.

        Returns:
            Result of the repair attempt.
        """
        from .genome import ExpressionLevel

        strategy = strategy or damage.recommended_strategy
        success = False
        details = ""

        if strategy == RepairStrategy.ROLLBACK:
            # Extract gene name from location
            gene_name = self._extract_gene_name(damage)
            if gene_name and genome.rollback_mutation(gene_name):
                success = True
                details = f"Rolled back last mutation on '{gene_name}'"
            else:
                details = f"Rollback failed for '{gene_name or damage.location}'"

        elif strategy == RepairStrategy.RE_EXPRESS:
            gene_name = self._extract_gene_name(damage)
            if gene_name:
                gene = genome.get_gene(gene_name)
                if gene:
                    target_level = gene.default_expression
                    genome.set_expression(gene_name, target_level, "dna_repair")
                    success = True
                    details = (
                        f"Reset expression of '{gene_name}' to "
                        f"{target_level.name}"
                    )
                else:
                    details = f"Gene '{gene_name}' not found"
            else:
                details = f"Cannot extract gene name from {damage.location}"

        elif strategy == RepairStrategy.EPIGENETIC_PATCH:
            # Store the damage as a lesson rather than modifying state
            self._store_repair_marker(
                RepairResult(
                    success=True,
                    strategy_used=strategy,
                    damage=damage,
                    details=f"Logged damage at {damage.location} as epigenetic marker",
                )
            )
            success = True
            details = (
                f"Stored repair lesson for {damage.corruption_type.value} "
                f"at {damage.location}"
            )

        elif strategy == RepairStrategy.CHECKPOINT_RESTORE:
            # Find the checkpoint that matches and restore all expression
            for cp in self._checkpoints.values():
                cp_expr = cp.expression_dict
                for gene_name, level_value in cp_expr.items():
                    if gene_name in genome._expression:
                        genome.set_expression(
                            gene_name,
                            ExpressionLevel(level_value),
                            "checkpoint_restore",
                        )
                success = True
                details = f"Restored expression state from checkpoint {cp.checkpoint_id}"
                break
            if not success:
                details = "No checkpoint available for restore"

        result = RepairResult(
            success=success,
            strategy_used=strategy,
            damage=damage,
            details=details,
        )
        self._repair_history.append(result)

        if success:
            self._store_repair_marker(result)

        if not self.silent:
            status = "\u2705" if success else "\u274c"
            print(
                f"\U0001f9ec [DNARepair] {status} {strategy.value}: {details}"
            )

        return result

    # ------------------------------------------------------------------
    # Certify
    # ------------------------------------------------------------------

    def certify(
        self,
        genome: Genome,
        checkpoint: StateCheckpoint,
    ) -> Certificate:
        """Issue a state integrity certificate.

        The certificate's ``verify()`` re-checks that the genome hash
        and gene count match the checkpoint.

        Args:
            genome: Current genome state.
            checkpoint: Reference checkpoint to verify against.

        Returns:
            A :class:`Certificate` with theorem ``state_integrity_verified``.
        """
        return Certificate(
            theorem="state_integrity_verified",
            parameters={
                "genome_hash": genome.get_hash(),
                "checkpoint_hash": checkpoint.genome_hash,
                "gene_count": len(genome._genes),
                "checkpoint_gene_count": checkpoint.gene_count,
            },
            conclusion=(
                "Genome state matches checkpoint with no detected corruption"
            ),
            source="DNARepair.certify",
            _verify_fn=_verify_state_integrity,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get DNA repair statistics."""
        return {
            "checkpoints": len(self._checkpoints),
            "scans_performed": self._scan_count,
            "repairs_attempted": len(self._repair_history),
            "repairs_successful": len(
                [r for r in self._repair_history if r.success]
            ),
            "repair_history": [
                {
                    "strategy": r.strategy_used.value,
                    "corruption": r.damage.corruption_type.value,
                    "location": r.damage.location,
                    "success": r.success,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self._repair_history
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_gene_name(damage: DamageReport) -> str | None:
        """Extract gene name from a damage report's location field."""
        # Locations like "expression:model" or "gene:temperature"
        if ":" in damage.location:
            parts = damage.location.split(":", 1)
            if parts[0] in ("expression", "gene"):
                return parts[1]
        return None

    def _store_repair_marker(self, result: RepairResult) -> None:
        """Store a successful repair as an epigenetic marker."""
        if self._histone_store is None:
            return
        self._histone_store.methylate(
            lesson=(
                f"Repaired {result.damage.corruption_type.value} at "
                f"{result.damage.location}: {result.details}"
            ),
            tags=[
                "repair",
                result.damage.corruption_type.value,
                result.strategy_used.value,
            ],
            context=f"DNARepair scan #{self._scan_count}",
        )
