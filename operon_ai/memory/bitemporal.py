"""
Bi-Temporal Memory: Append-Only Factual Memory with Two Time Axes.
=================================================================

Tracks both **valid time** (when a fact is true in the world) and
**record time** (when the system learned about it).  Corrections are
append-only — old records are closed, never mutated.

This mirrors the way biological memory works at the cellular level:
a neuron does not erase a synapse when new information arrives;
instead, new synaptic connections form alongside the old ones,
and the relative strength determines which "version" the organism
acts on.  Bi-temporal memory applies the same principle to
structured facts.

Features:
- Immutable fact records with dual time axes
- Append-only corrections that preserve full history
- Point-in-time retrieval: valid-time, record-time, or both
- History / diff / timeline APIs for auditing
- Belief-state reconstruction at any (valid, record) coordinate

Usage::

    from operon_ai import BiTemporalMemory

    mem = BiTemporalMemory()

    fact = mem.record_fact(
        subject="client:42",
        predicate="risk_tier",
        value="medium",
        valid_from=day1,
        recorded_from=day3,
        source="crm",
    )

    correction = mem.correct_fact(
        old_fact_id=fact.fact_id,
        value="high",
        valid_from=day1,
        recorded_from=day5,
        source="manual_review",
    )

    # What is true now?
    mem.retrieve_valid_at(subject="client:42", at=day2)

    # What did the system know on day 4?
    mem.retrieve_known_at(subject="client:42", at=day4)

    # What did the system *believe* was true on day 2,
    # given only what it knew by day 4?
    mem.retrieve_belief_state(at_valid=day2, at_record=day4)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BiTemporalFact:
    """A single immutable fact with valid-time and record-time intervals."""

    fact_id: str
    subject: str
    predicate: str
    value: Any
    valid_from: datetime
    valid_to: datetime | None
    recorded_from: datetime
    recorded_to: datetime | None
    source: str
    confidence: float = 1.0
    tags: tuple[str, ...] = ()
    supersedes: str | None = None


@dataclass(frozen=True)
class BiTemporalQuery:
    """Filter specification for point-in-time queries."""

    subject: str | None = None
    predicate: str | None = None
    at_valid: datetime | None = None
    at_record: datetime | None = None


@dataclass(frozen=True)
class FactSnapshot:
    """Result container for a point-in-time query."""

    facts: tuple[BiTemporalFact, ...]
    query: BiTemporalQuery
    snapshot_time: datetime


@dataclass(frozen=True)
class CorrectionResult:
    """Result of correcting an existing fact."""

    old_fact: BiTemporalFact
    new_fact: BiTemporalFact
    correction_time: datetime


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class BiTemporalMemory:
    """Append-only bi-temporal fact store.

    Internal indexes are maintained automatically on every write
    operation.  All reads are pure filters over the fact list —
    no mutation occurs during retrieval.
    """

    _facts: list[BiTemporalFact] = field(default_factory=list)
    _by_id: dict[str, BiTemporalFact] = field(default_factory=dict)
    _by_subject: dict[str, list[str]] = field(default_factory=dict)

    # -- private helpers ---------------------------------------------------

    def _append(self, fact: BiTemporalFact) -> None:
        self._facts.append(fact)
        self._by_id[fact.fact_id] = fact
        self._by_subject.setdefault(fact.subject, []).append(fact.fact_id)

    def _replace_fact(self, fact_id: str, new_version: BiTemporalFact) -> None:
        self._by_id[fact_id] = new_version
        for i, f in enumerate(self._facts):
            if f.fact_id == fact_id:
                self._facts[i] = new_version
                break

    # -- write semantics ---------------------------------------------------

    def record_fact(
        self,
        subject: str,
        predicate: str,
        value: Any,
        valid_from: datetime,
        recorded_from: datetime,
        source: str,
        confidence: float = 1.0,
        tags: tuple[str, ...] = (),
        valid_to: datetime | None = None,
    ) -> BiTemporalFact:
        """Insert a new active fact record."""
        fact = BiTemporalFact(
            fact_id=str(uuid.uuid4())[:8],
            subject=subject,
            predicate=predicate,
            value=value,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_from=recorded_from,
            recorded_to=None,
            source=source,
            confidence=confidence,
            tags=tags,
            supersedes=None,
        )
        self._append(fact)
        return fact

    def correct_fact(
        self,
        old_fact_id: str,
        value: Any,
        valid_from: datetime,
        recorded_from: datetime,
        source: str,
        confidence: float = 1.0,
        tags: tuple[str, ...] = (),
        valid_to: datetime | None = None,
    ) -> CorrectionResult:
        """Correct a fact: close the old record and append a new one."""
        old = self._by_id.get(old_fact_id)
        if old is None:
            raise KeyError(f"No fact with id {old_fact_id!r}")

        closed_old = replace(old, recorded_to=recorded_from)
        self._replace_fact(old_fact_id, closed_old)

        new_fact = BiTemporalFact(
            fact_id=str(uuid.uuid4())[:8],
            subject=old.subject,
            predicate=old.predicate,
            value=value,
            valid_from=valid_from,
            valid_to=valid_to,
            recorded_from=recorded_from,
            recorded_to=None,
            source=source,
            confidence=confidence,
            tags=tags,
            supersedes=old_fact_id,
        )
        self._append(new_fact)

        return CorrectionResult(
            old_fact=closed_old,
            new_fact=new_fact,
            correction_time=recorded_from,
        )

    def invalidate_fact(
        self,
        fact_id: str,
        recorded_to: datetime | None = None,
    ) -> BiTemporalFact:
        """Mark a fact as no longer active knowledge."""
        old = self._by_id.get(fact_id)
        if old is None:
            raise KeyError(f"No fact with id {fact_id!r}")
        closed = replace(old, recorded_to=recorded_to or datetime.now())
        self._replace_fact(fact_id, closed)
        return closed

    # -- private predicates ------------------------------------------------

    @staticmethod
    def _matches(
        fact: BiTemporalFact,
        subject: str | None,
        predicate: str | None,
    ) -> bool:
        if subject is not None and fact.subject != subject:
            return False
        if predicate is not None and fact.predicate != predicate:
            return False
        return True

    @staticmethod
    def _valid_at(fact: BiTemporalFact, at: datetime) -> bool:
        if fact.valid_from > at:
            return False
        if fact.valid_to is not None and at >= fact.valid_to:
            return False
        return True

    @staticmethod
    def _recorded_at(fact: BiTemporalFact, at: datetime) -> bool:
        if fact.recorded_from > at:
            return False
        if fact.recorded_to is not None and at >= fact.recorded_to:
            return False
        return True

    # -- point-in-time retrieval -------------------------------------------

    def retrieve_valid_at(
        self,
        at: datetime,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> list[BiTemporalFact]:
        """Facts currently valid at world-time *at* (active records only)."""
        return [
            f for f in self._facts
            if self._matches(f, subject, predicate)
            and self._valid_at(f, at)
            and f.recorded_to is None
        ]

    def retrieve_known_at(
        self,
        at: datetime,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> list[BiTemporalFact]:
        """Facts the system had recorded by record-time *at*."""
        return [
            f for f in self._facts
            if self._matches(f, subject, predicate)
            and self._recorded_at(f, at)
        ]

    def retrieve_belief_state(
        self,
        at_valid: datetime,
        at_record: datetime,
    ) -> list[BiTemporalFact]:
        """Reconstruct the system's belief at a (valid, record) coordinate."""
        return [
            f for f in self._facts
            if self._valid_at(f, at_valid)
            and self._recorded_at(f, at_record)
        ]

    # -- history / diff / timeline -----------------------------------------

    def history(
        self,
        subject: str,
        predicate: str | None = None,
    ) -> list[BiTemporalFact]:
        """All facts for *subject* (including closed), sorted by recorded_from."""
        return sorted(
            [f for f in self._facts if self._matches(f, subject, predicate)],
            key=lambda f: f.recorded_from,
        )

    def diff_between(
        self,
        t1: datetime,
        t2: datetime,
        axis: str = "valid",
    ) -> list[BiTemporalFact]:
        """Facts present at *t2* but not at *t1* on the given axis."""
        if axis == "valid":
            ids_t1 = {f.fact_id for f in self._facts if self._valid_at(f, t1) and f.recorded_to is None}
            ids_t2 = {f.fact_id for f in self._facts if self._valid_at(f, t2) and f.recorded_to is None}
        elif axis == "record":
            ids_t1 = {f.fact_id for f in self._facts if self._recorded_at(f, t1)}
            ids_t2 = {f.fact_id for f in self._facts if self._recorded_at(f, t2)}
        else:
            raise ValueError(f"axis must be 'valid' or 'record', got {axis!r}")
        new_ids = ids_t2 - ids_t1
        return [f for f in self._facts if f.fact_id in new_ids]

    def timeline_for(
        self,
        subject: str,
        predicate: str | None = None,
    ) -> list[BiTemporalFact]:
        """All facts for *subject* sorted by valid_from (full world-time timeline)."""
        return sorted(
            [f for f in self._facts if self._matches(f, subject, predicate)],
            key=lambda f: f.valid_from,
        )
