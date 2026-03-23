"""Social learning — cross-organism template sharing with epistemic vigilance.

Enables organisms to share successful PatternTemplates across PatternLibrary
instances. Trust scoring (epistemic vigilance) modulates adoption decisions
based on the track record of shared templates.

Biological analogy: horizontal gene transfer (HGT) in bacteria — organisms
exchange genetic material through conjugation. Epistemic vigilance determines
whether foreign material is incorporated or rejected.

References:
  Dupoux, LeCun & Malik (arXiv:2603.15381) — Appendix B.1: social learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..patterns.repository import (
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerExchange:
    """Bundle of templates and run records offered by a peer organism."""

    peer_id: str
    templates: tuple[PatternTemplate, ...]
    records: tuple[PatternRunRecord, ...]
    exported_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class AdoptionResult:
    """Outcome of importing templates from a peer."""

    peer_id: str
    adopted_template_ids: tuple[str, ...]
    rejected_template_ids: tuple[str, ...]
    trust_score_used: float
    trust_score_after: float


@dataclass(frozen=True)
class AdoptionOutcome:
    """Records whether an adopted template succeeded for us."""

    peer_id: str
    template_id: str
    success: bool
    recorded_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Trust registry (epistemic vigilance)
# ---------------------------------------------------------------------------


@dataclass
class TrustRegistry:
    """Per-peer trust scoring based on adopted template outcomes.

    Uses exponential moving average (EMA) over adoption outcomes.
    Trust increases on success, decreases on failure, with recent
    outcomes weighted more heavily.
    """

    default_trust: float = 0.5
    decay_alpha: float = 0.3  # EMA smoothing (higher = more recent-weighted)
    min_trust_to_adopt: float = 0.2

    _scores: dict[str, float] = field(default_factory=dict)
    _outcomes: list[AdoptionOutcome] = field(default_factory=list)

    def trust_score(self, peer_id: str) -> float:
        """Return current trust for a peer (default_trust if unknown)."""
        return self._scores.get(peer_id, self.default_trust)

    def record_outcome(
        self,
        peer_id: str,
        template_id: str,
        success: bool,
    ) -> float:
        """Record adoption outcome; return updated trust score."""
        self._outcomes.append(AdoptionOutcome(
            peer_id=peer_id,
            template_id=template_id,
            success=success,
        ))
        old = self._scores.get(peer_id, self.default_trust)
        outcome_val = 1.0 if success else 0.0
        new = self.decay_alpha * outcome_val + (1.0 - self.decay_alpha) * old
        self._scores[peer_id] = new
        return new

    def is_trusted(self, peer_id: str) -> bool:
        """Check if peer's trust exceeds min_trust_to_adopt."""
        return self.trust_score(peer_id) >= self.min_trust_to_adopt

    def peer_rankings(self) -> list[dict[str, Any]]:
        """Return peers ranked by trust score (descending)."""
        ranked = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        return [{"peer_id": pid, "trust": score} for pid, score in ranked]

    def summary(self) -> dict[str, Any]:
        """Return registry statistics."""
        return {
            "peer_count": len(self._scores),
            "total_outcomes": len(self._outcomes),
            "mean_trust": (
                sum(self._scores.values()) / len(self._scores)
                if self._scores else self.default_trust
            ),
        }


# ---------------------------------------------------------------------------
# Social learning
# ---------------------------------------------------------------------------

_ADOPTION_THRESHOLD = 0.3  # effective_score must exceed this to adopt


@dataclass
class SocialLearning:
    """Cross-organism template sharing with trust-weighted adoption.

    Wraps a PatternLibrary and adds peer-exchange semantics.
    Social learning is opt-in — instantiate this class only when
    cross-organism sharing is desired.
    """

    organism_id: str
    library: PatternLibrary
    trust: TrustRegistry = field(default_factory=TrustRegistry)
    silent: bool = False

    _adopted_from: dict[str, str] = field(default_factory=dict)
    _exchange_log: list[PeerExchange] = field(default_factory=list)

    def export_templates(
        self,
        *,
        min_success_rate: float = 0.6,
        min_runs: int = 1,
        tags: tuple[str, ...] | None = None,
    ) -> PeerExchange:
        """Export templates that meet success criteria."""
        exported: list[PatternTemplate] = []
        exported_records: list[PatternRunRecord] = []

        for tmpl in self.library._templates.values():
            if tags is not None and not set(tags).issubset(set(tmpl.tags)):
                continue
            sr = self.library.success_rate(tmpl.template_id)
            if sr is None:
                continue
            runs = [r for r in self.library._records if r.template_id == tmpl.template_id]
            if len(runs) < min_runs:
                continue
            if sr < min_success_rate:
                continue
            exported.append(tmpl)
            exported_records.extend(runs)

        return PeerExchange(
            peer_id=self.organism_id,
            templates=tuple(exported),
            records=tuple(exported_records),
        )

    def import_from_peer(
        self,
        exchange: PeerExchange,
        *,
        trust_override: float | None = None,
    ) -> AdoptionResult:
        """Adopt templates from a peer, weighted by trust."""
        peer_id = exchange.peer_id
        trust = trust_override if trust_override is not None else self.trust.trust_score(peer_id)
        self._exchange_log.append(exchange)

        if trust < self.trust.min_trust_to_adopt:
            return AdoptionResult(
                peer_id=peer_id,
                adopted_template_ids=(),
                rejected_template_ids=tuple(t.template_id for t in exchange.templates),
                trust_score_used=trust,
                trust_score_after=self.trust.trust_score(peer_id),
            )

        adopted: list[str] = []
        rejected: list[str] = []

        # Compute per-template success rate from exchange records
        for tmpl in exchange.templates:
            tmpl_records = [r for r in exchange.records if r.template_id == tmpl.template_id]
            if tmpl_records:
                peer_sr = sum(1 for r in tmpl_records if r.success) / len(tmpl_records)
            else:
                peer_sr = 0.5

            effective_score = peer_sr * trust
            if effective_score >= _ADOPTION_THRESHOLD:
                self.library.register_template(tmpl)
                self._adopted_from[tmpl.template_id] = peer_id
                adopted.append(tmpl.template_id)
            else:
                rejected.append(tmpl.template_id)

        return AdoptionResult(
            peer_id=peer_id,
            adopted_template_ids=tuple(adopted),
            rejected_template_ids=tuple(rejected),
            trust_score_used=trust,
            trust_score_after=self.trust.trust_score(peer_id),
        )

    def record_adoption_outcome(
        self,
        template_id: str,
        success: bool,
    ) -> float | None:
        """Record whether an adopted template succeeded for us.

        Returns updated trust score, or None if template wasn't from a peer.
        """
        peer_id = self._adopted_from.get(template_id)
        if peer_id is None:
            return None
        return self.trust.record_outcome(peer_id, template_id, success)

    def get_provenance(self, template_id: str) -> str | None:
        """Return peer_id that shared a template, or None if locally created."""
        return self._adopted_from.get(template_id)

    def summary(self) -> dict[str, Any]:
        """Return social learning statistics."""
        return {
            "organism_id": self.organism_id,
            "adopted_count": len(self._adopted_from),
            "exchanges_received": len(self._exchange_log),
            "trust": self.trust.summary(),
        }
