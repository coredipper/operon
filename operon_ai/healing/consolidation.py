"""Sleep consolidation — post-batch distillation of operational history.

Extends AutophagyDaemon with replay, compression, counterfactual analysis,
and histone promotion. Mirrors neuroscientific sleep-time memory consolidation.

References:
  Dupoux, LeCun & Malik (arXiv:2603.15381) — imagination-based learning
  during rest via System M meta-control
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..memory.bitemporal import BiTemporalFact, BiTemporalMemory
from ..memory.episodic import EpisodicMemory, MemoryTier
from ..patterns.repository import (
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    TaskFingerprint,
)
from ..state.histone import HistoneStore, MarkerType, MarkerStrength
from .autophagy_daemon import AutophagyDaemon, PruneResult


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterfactualResult:
    """Result of replaying a past run record with updated bi-temporal facts."""

    record: PatternRunRecord
    corrections_found: tuple[BiTemporalFact, ...]
    affected_stages: tuple[str, ...]
    outcome_would_change: bool
    reasoning: str


@dataclass(frozen=True)
class ConsolidationResult:
    """Outcome of a sleep consolidation cycle."""

    templates_created: int
    memories_promoted: int
    histone_promotions: int
    counterfactual_results: tuple[CounterfactualResult, ...]
    prune_result: PruneResult | None
    duration_ms: float


# ---------------------------------------------------------------------------
# Counterfactual replay
# ---------------------------------------------------------------------------


def counterfactual_replay(
    record: PatternRunRecord,
    bitemporal: BiTemporalMemory,
    run_time: datetime | None = None,
    now: datetime | None = None,
) -> CounterfactualResult:
    """Replay a past run record against updated bi-temporal facts.

    Static analysis — no re-execution. Uses diff_between() to find
    corrections since the run, then checks if any affect stages in
    the record's template.
    """
    run_time = run_time or record.recorded_at
    now = now or datetime.now()

    corrections = bitemporal.diff_between(run_time, now, axis="record")
    if not corrections:
        return CounterfactualResult(
            record=record,
            corrections_found=(),
            affected_stages=(),
            outcome_would_change=False,
            reasoning="No corrections found since run time.",
        )

    # Match corrections to stage names via subject/predicate
    stage_names = {
        intv.get("stage_name", "")
        for intv in record.interventions
        if isinstance(intv, dict)
    }
    # Also match against fingerprint roles
    role_names = set(record.fingerprint.required_roles) if record.fingerprint else set()
    match_targets = stage_names | role_names

    affected: list[str] = []
    for fact in corrections:
        for target in match_targets:
            if target and (target in fact.subject or target in fact.predicate):
                affected.append(target)

    affected_unique = tuple(sorted(set(affected)))
    outcome_would_change = len(affected_unique) > 0

    if outcome_would_change:
        reasoning = (
            f"{len(corrections)} correction(s) found since run. "
            f"Affected stages/roles: {', '.join(affected_unique)}. "
            f"Outcome may have differed with updated facts."
        )
    else:
        reasoning = (
            f"{len(corrections)} correction(s) found since run, "
            f"but none match stage names or roles in this template."
        )

    return CounterfactualResult(
        record=record,
        corrections_found=tuple(corrections),
        affected_stages=affected_unique,
        outcome_would_change=outcome_would_change,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# SleepConsolidation
# ---------------------------------------------------------------------------


@dataclass
class SleepConsolidation:
    """Post-batch consolidation cycle extending AutophagyDaemon.

    Composes existing components via delegation, not inheritance:
      - AutophagyDaemon: context pruning
      - PatternLibrary: run records and template registration
      - EpisodicMemory: tiered memory with promotion
      - HistoneStore: epigenetic marks with decay
      - BiTemporalMemory (optional): counterfactual replay
    """

    daemon: AutophagyDaemon
    pattern_library: PatternLibrary
    episodic_memory: EpisodicMemory
    histone_store: HistoneStore
    bitemporal_memory: BiTemporalMemory | None = None

    # Tuning
    min_success_rate_for_template: float = 0.7
    min_access_count_for_promotion: int = 3
    acetylation_promotion_threshold: int = 5
    silent: bool = False

    def consolidate(
        self,
        context: str = "",
        max_tokens: int = 8000,
    ) -> ConsolidationResult:
        """Run the full sleep consolidation cycle.

        1. Prune stale context via AutophagyDaemon
        2. Replay successful run records, promote in EpisodicMemory
        3. Compress recurring patterns into new PatternTemplates
        4. Run counterfactual replay on corrected bi-temporal facts
        5. Promote important histone marks (ACETYLATION -> METHYLATION)
        """
        start = datetime.now()

        # 1. Prune
        prune_result: PruneResult | None = None
        if context:
            _, prune_result = self.daemon.check_and_prune(context, max_tokens)

        # 2. Replay
        memories_promoted = self._replay_successful_patterns()

        # 3. Compress
        templates_created = self._compress_to_templates()

        # 4. Counterfactual
        counterfactuals = self._run_counterfactual_replay()

        # 5. Promote histone marks
        histone_promotions = self._promote_histone_marks()

        duration_ms = (datetime.now() - start).total_seconds() * 1000

        return ConsolidationResult(
            templates_created=templates_created,
            memories_promoted=memories_promoted,
            histone_promotions=histone_promotions,
            counterfactual_results=tuple(counterfactuals),
            prune_result=prune_result,
            duration_ms=duration_ms,
        )

    def _replay_successful_patterns(self) -> int:
        """Replay successful run records and promote in episodic memory."""
        promoted = 0
        seen_templates: dict[str, int] = defaultdict(int)

        for record in self.pattern_library._records:
            if not record.success:
                continue
            seen_templates[record.template_id] += 1
            content = (
                f"Successful run of template {record.template_id}: "
                f"latency={record.latency_ms:.0f}ms, tokens={record.tokens_used}"
            )
            entry = self.episodic_memory.store(
                content=content,
                tier=MemoryTier.WORKING,
                histone_marks={"reliability": 1.0, "importance": 0.7},
            )
            # Promote to EPISODIC if template seen multiple times
            if seen_templates[record.template_id] >= self.min_access_count_for_promotion:
                self.episodic_memory.promote(entry.id, MemoryTier.EPISODIC)
                promoted += 1

        return promoted

    def _compress_to_templates(self) -> int:
        """Create new distilled templates for high-performing patterns."""
        created = 0
        # Group records by template_id
        by_template: dict[str, list[PatternRunRecord]] = defaultdict(list)
        for record in self.pattern_library._records:
            by_template[record.template_id].append(record)

        for template_id, records in by_template.items():
            sr = self.pattern_library.success_rate(template_id)
            if sr is None or sr < self.min_success_rate_for_template:
                continue
            if len(records) < self.min_access_count_for_promotion:
                continue
            # Check if a consolidated template already exists
            existing = self.pattern_library.get_template(template_id)
            if existing is None:
                continue
            # Create a distilled version with a "consolidated" tag
            if "consolidated" in existing.tags:
                continue  # Already consolidated

            consolidated = PatternTemplate(
                template_id=f"{template_id}_c",
                name=f"{existing.name} (consolidated)",
                topology=existing.topology,
                stage_specs=existing.stage_specs,
                intervention_policy=existing.intervention_policy,
                fingerprint=existing.fingerprint,
                tags=existing.tags + ("consolidated",),
            )
            self.pattern_library.register_template(consolidated)
            created += 1

        return created

    def _run_counterfactual_replay(self) -> list[CounterfactualResult]:
        """Run counterfactual analysis on recent run records."""
        if self.bitemporal_memory is None:
            return []
        results: list[CounterfactualResult] = []
        for record in self.pattern_library._records:
            cr = counterfactual_replay(record, self.bitemporal_memory)
            if cr.corrections_found:
                results.append(cr)
        return results

    def _promote_histone_marks(self) -> int:
        """Promote frequently-accessed ACETYLATION marks to METHYLATION."""
        promoted = 0
        to_promote: list[tuple[str, str]] = []  # (hash, content) pairs

        for marker_hash, marker in list(self.histone_store._markers.items()):
            if (
                marker.marker_type == MarkerType.ACETYLATION
                and marker.access_count >= self.acetylation_promotion_threshold
                and not marker.is_expired()
            ):
                to_promote.append((marker_hash, marker.content))

        for marker_hash, content in to_promote:
            self.histone_store.remove_marker(marker_hash)
            self.histone_store.add_marker(
                content=content,
                marker_type=MarkerType.METHYLATION,
                strength=MarkerStrength.PERMANENT,
                tags=["consolidated", "promoted_from_acetylation"],
            )
            promoted += 1

        return promoted
