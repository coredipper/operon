"""Memory bridge -- import external memory formats into BiTemporalMemory.

Bridges AnimaWorks episodic memory and DeerFlow session/vector memory into
Operon's append-only bi-temporal fact store.  Like the adapters in
``operon_ai.memory.adapters``, these functions create new facts without
modifying the source data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from ..memory.bitemporal import BiTemporalFact, BiTemporalMemory


def _content_hash(data: Any) -> str:
    """Stable 8-char hash from content for unique subject keys."""
    return hashlib.sha256(str(data).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime.

    * Naive timestamps (no tzinfo) are treated as UTC.
    * ``None`` returns the current UTC time.
    """
    if ts is None:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# AnimaWorks bridge
# ---------------------------------------------------------------------------

def bridge_animaworks_memory(
    entries: list[dict],
    target: BiTemporalMemory,
    subject_prefix: str = "animaworks",
) -> list[BiTemporalFact]:
    """Bridge AnimaWorks memory entries into bi-temporal facts.

    Each entry becomes a fact with:
      - subject: ``"{prefix}:{entry['id']}"``
      - predicate: entry type (e.g. ``"episodic"``)
      - value: entry content string
      - valid_from / recorded_from: parsed entry timestamp
      - source: ``"animaworks:{source_agent}"``

    Returns the list of created facts.
    """
    created: list[BiTemporalFact] = []
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp"))
        fact = target.record_fact(
            subject=f"{subject_prefix}:{entry['id']}",
            predicate=entry["type"],
            value=entry["content"],
            valid_from=ts,
            recorded_from=ts,
            source=f"animaworks:{entry.get('source_agent', 'unknown')}",
        )
        created.append(fact)
    return created


# ---------------------------------------------------------------------------
# DeerFlow bridge
# ---------------------------------------------------------------------------

def bridge_deerflow_memory(
    session_memory: list[dict],
    vector_entries: list[dict],
    target: BiTemporalMemory,
    subject_prefix: str = "deerflow",
    *,
    session_id: str,
) -> list[BiTemporalFact]:
    """Bridge DeerFlow session messages and vector-store entries into facts.

    Session messages become facts with:
      - subject: ``"{prefix}:session:{idx}"``
      - predicate: message role
      - value: message content

    Vector entries become facts with:
      - subject: ``"{prefix}:vector:{entry['id']}"``
      - predicate: ``"vector_entry"``
      - value: entry content

    Returns the combined list of created facts.
    """
    created: list[BiTemporalFact] = []

    for idx, msg in enumerate(session_memory):
        ts = _parse_timestamp(msg.get("timestamp"))
        # Subject embeds session_id + index directly — deterministic and collision-free.
        fact = target.record_fact(
            subject=f"{subject_prefix}:session:{session_id}:{idx}",
            predicate=msg["role"],
            value=msg["content"],
            valid_from=ts,
            recorded_from=ts,
            source=f"deerflow:session",
        )
        created.append(fact)

    for entry in vector_entries:
        ts = _parse_timestamp(entry.get("inserted_at"))
        fact = target.record_fact(
            subject=f"{subject_prefix}:vector:{entry['id']}",
            predicate="vector_entry",
            value=entry["content"],
            valid_from=ts,
            recorded_from=ts,
            source=f"deerflow:vector",
        )
        created.append(fact)

    return created
