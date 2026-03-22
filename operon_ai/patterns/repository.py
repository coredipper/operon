"""Pattern repository — evolutionary memory of successful collaboration patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class TaskFingerprint:
    """Characterizes a task for pattern matching."""

    task_shape: str  # "sequential", "parallel", "mixed"
    tool_count: int
    subtask_count: int
    required_roles: tuple[str, ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatternTemplate:
    """Reusable collaboration pattern blueprint."""

    template_id: str
    name: str
    topology: str  # "single_worker", "reviewer_gate", "specialist_swarm", "skill_organism"
    stage_specs: tuple[dict[str, Any], ...]
    intervention_policy: dict[str, Any]
    fingerprint: TaskFingerprint
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PatternRunRecord:
    """Outcome record for a pattern execution."""

    record_id: str
    template_id: str
    fingerprint: TaskFingerprint
    success: bool
    latency_ms: float
    tokens_used: int
    interventions: tuple[dict[str, Any], ...] = ()
    notes: str = ""
    recorded_at: datetime = field(default_factory=datetime.now)


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Jaccard similarity between two string tuples."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


@dataclass
class PatternLibrary:
    """In-memory registry of reusable collaboration pattern templates."""

    _templates: dict[str, PatternTemplate] = field(default_factory=dict)
    _records: list[PatternRunRecord] = field(default_factory=list)

    # -- Template CRUD ---------------------------------------------------

    def register_template(self, template: PatternTemplate) -> None:
        """Add or replace a template in the library."""
        self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> PatternTemplate | None:
        """Retrieve a single template by ID."""
        return self._templates.get(template_id)

    def retrieve_templates(
        self,
        *,
        topology: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> list[PatternTemplate]:
        """Retrieve templates filtered by topology and/or tags."""
        out: list[PatternTemplate] = []
        for t in self._templates.values():
            if topology is not None and t.topology != topology:
                continue
            if tags is not None and not set(tags).issubset(set(t.tags)):
                continue
            out.append(t)
        return out

    # -- Run records ------------------------------------------------------

    def record_run(self, record: PatternRunRecord) -> None:
        """Record the outcome of a pattern execution."""
        self._records.append(record)

    def success_rate(self, template_id: str) -> float | None:
        """Return success rate for a template, or None if no records."""
        runs = [r for r in self._records if r.template_id == template_id]
        if not runs:
            return None
        return sum(1 for r in runs if r.success) / len(runs)

    # -- Ranking ----------------------------------------------------------

    def top_templates_for(
        self,
        fingerprint: TaskFingerprint,
        *,
        limit: int = 5,
    ) -> list[tuple[PatternTemplate, float]]:
        """Return ranked (template, score) pairs for a given fingerprint.

        Scoring weights:
          0.30 — task_shape exact match
          0.15 — tool_count proximity
          0.15 — subtask_count proximity
          0.20 — required_roles Jaccard overlap
          0.10 — tag Jaccard overlap
          0.10 — historical success rate
        """
        scored: list[tuple[PatternTemplate, float]] = []
        for t in self._templates.values():
            fp = t.fingerprint
            shape = 1.0 if fp.task_shape == fingerprint.task_shape else 0.0
            tool_prox = 1.0 / (1.0 + abs(fp.tool_count - fingerprint.tool_count))
            sub_prox = 1.0 / (1.0 + abs(fp.subtask_count - fingerprint.subtask_count))
            role_sim = _jaccard(fp.required_roles, fingerprint.required_roles)
            tag_sim = _jaccard(t.tags, fingerprint.tags)
            sr = self.success_rate(t.template_id)
            sr_score = sr if sr is not None else 0.5

            score = (
                0.30 * shape
                + 0.15 * tool_prox
                + 0.15 * sub_prox
                + 0.20 * role_sim
                + 0.10 * tag_sim
                + 0.10 * sr_score
            )
            scored.append((t, round(score, 4)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def make_id() -> str:
        """Generate a short unique ID."""
        return uuid4().hex[:8]

    def summary(self) -> dict[str, Any]:
        """Return library statistics."""
        topologies: dict[str, int] = {}
        for t in self._templates.values():
            topologies[t.topology] = topologies.get(t.topology, 0) + 1
        return {
            "template_count": len(self._templates),
            "record_count": len(self._records),
            "topologies": topologies,
        }
