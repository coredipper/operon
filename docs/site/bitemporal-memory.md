# Bi-Temporal Memory

Operon's `BiTemporalMemory` tracks facts along two independent time axes:

- **Valid time** — when a fact is true in the world.
- **Record time** — when the system learned about it.

Unlike `HistoneStore` (epigenetic marks on memories) or `EpisodicMemory` (tiered decay), bi-temporal memory is append-only — corrections close old records and create new ones, so the full history is always reconstructible.

## Why two time axes?

A single timestamp conflates two different questions:

1. "What is true now?" (valid-time query)
2. "What did the system believe at decision time?" (record-time query)

These diverge whenever facts are ingested with delay or corrected retroactively — which is the norm in compliance, healthcare, and finance workflows. Bi-temporal memory makes the divergence explicit and queryable.

## Quick start

```python
from datetime import datetime, timedelta
from operon_ai import BiTemporalMemory

mem = BiTemporalMemory()

day1 = datetime(2025, 1, 1)
day3 = day1 + timedelta(days=2)
day5 = day1 + timedelta(days=4)

# Record a fact: client risk tier is "medium" (valid day 1, ingested day 3)
fact = mem.record_fact("client:42", "risk_tier", "medium",
                       valid_from=day1, recorded_from=day3, source="crm")

# Correct it: actually "high", retroactive to day 1 (recorded day 5)
mem.correct_fact(fact.fact_id, "high",
                 valid_from=day1, recorded_from=day5, source="manual_review")
```

## Querying

Three retrieval methods correspond to three questions:

```python
day2 = day1 + timedelta(days=1)
day4 = day1 + timedelta(days=3)
day6 = day1 + timedelta(days=5)

# What is true at day 2? (current active records only)
mem.retrieve_valid_at(at=day2, subject="client:42")       # → "high"

# What had the system recorded by day 2?
mem.retrieve_known_at(at=day2, subject="client:42")        # → [] (not yet ingested)

# What did the system believe was true on day 2, given what it knew by day 4?
mem.retrieve_belief_state(at_valid=day2, at_record=day4)   # → "medium"

# Same question, but given what it knows by day 6 (after correction)?
mem.retrieve_belief_state(at_valid=day2, at_record=day6)   # → "high"
```

## History and audit

```python
# Full correction history for a subject, sorted by record time
mem.history("client:42")

# What changed between two points on either axis?
mem.diff_between(day4, day6, axis="record")

# World-time timeline, sorted by valid_from
mem.timeline_for("client:42")
```

## Append-only semantics

Corrections never mutate existing records. Instead:

1. The old record's `recorded_to` is set (closing it on the record-time axis).
2. A new record is appended with `supersedes` pointing to the old `fact_id`.

This means any past belief state is always reconstructible — the foundation for compliance auditing.

## Data model

| Type | Purpose |
|------|---------|
| `BiTemporalFact` | Immutable fact with dual time intervals (frozen dataclass) |
| `BiTemporalQuery` | Filter spec for point-in-time queries |
| `BiTemporalMemory` | Mutable store with append-only write semantics |
| `FactSnapshot` | Query result container |
| `CorrectionResult` | Result of `correct_fact()` — old fact, new fact, correction time |
| `SubstrateView` | Frozen read-only envelope of facts for stage handlers (v0.20.0) |

All types are exported from the top level: `from operon_ai import BiTemporalMemory, BiTemporalFact, SubstrateView, ...`

## SkillOrganism Integration

Since v0.20.0, `BiTemporalMemory` can serve as a shared substrate for `SkillOrganism` workflows. Pass `substrate=BiTemporalMemory()` to `skill_organism(...)`, then use per-stage hooks to read and write facts:

| `SkillStage` field | Purpose |
|---|---|
| `read_query` | Subject string or callable → `SubstrateView` injected before stage runs |
| `fact_extractor` | Callable → emits assert/correct/invalidate events after stage runs |
| `emit_output_fact` | Convenience flag: auto-records `(task, stage.name, output)` |
| `fact_tags` | Default tags applied to all facts emitted by this stage |

Handlers receive the `SubstrateView` as an additional argument (via arity-aware dispatch — existing handlers are unaffected). See [Skill Organisms](skill-organisms.md) for the full three-layer context model.

## Examples

- [`examples/69_bitemporal_memory.py`](../../examples/69_bitemporal_memory.py) — core valid-time vs record-time divergence with corrections
- [`examples/70_bitemporal_compliance_audit.py`](../../examples/70_bitemporal_compliance_audit.py) — multi-fact compliance audit with belief-state reconstruction
- [`examples/71_bitemporal_skill_organism.py`](../../examples/71_bitemporal_skill_organism.py) — multi-stage organism with substrate, belief-state reconstruction, and temporal diffs
- [Bi-Temporal Memory Explorer](https://huggingface.co/spaces/coredipper/operon-bitemporal) — interactive HuggingFace Space

## Relationship to other memory systems

| System | Time model | Mutation | Use case |
|--------|-----------|----------|----------|
| `HistoneStore` | Single timestamp | Mutable marks | Epigenetic metadata on retrieval |
| `EpisodicMemory` | Single timestamp | Mutable (decay, promote) | Tiered memory with relevance scoring |
| `BiTemporalMemory` | Dual timestamps | Append-only | Auditable fact tracking with corrections |

The three systems are intentionally decoupled. Integration bridges (converting histone marks or episodic memories into bi-temporal facts) are planned for future work.
