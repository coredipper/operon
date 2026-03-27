# Skill Organisms

`skill_organism(...)` is the current practical runtime for building one structured workflow out of multiple stages.

## Why It Exists

A lot of real multi-agent workflows are not “many agents everywhere.”

They look more like:

- normalize input
- classify or route cheaply
- do the ambiguous work with a stronger model
- attach telemetry, review, or other runtime concerns without rewriting the whole workflow

That is the use case `skill_organism(...)` is meant to serve.

## Core Pieces

- `SkillStage`
- `SkillOrganism`
- `TelemetryProbe`

Stages can be:

- deterministic handlers
- provider-bound agents using a fast nucleus
- provider-bound agents using a deep nucleus

## Practical Pattern

Typical workflow:

1. intake or normalization stage
2. cheap routing/classification stage
3. deeper planning or synthesis stage
4. optional attached telemetry or review

This lets one workflow behave more like a composed organism than a single prompt or an unstructured chain of scripts.

## Three-Layer Context Model

Stages within an organism have access to three layers of context, each with different lifetime and mutability:

1. **Topology** — the wiring diagram and observation structure. Structural and static within a single run.
2. **Ephemeral** — the `shared_state` dictionary. Carries routing hints, counters, and stage outputs. Mutable, not historically reconstructible.
3. **Bi-temporal** — an optional `BiTemporalMemory` substrate. Carries durable factual knowledge with dual time axes. Append-only and fully auditable.

### Ephemeral: shared_state

`shared_state` is still useful for lightweight orchestration data — routing labels, counters, temporary outputs.

### Bi-temporal: substrate

When you pass `substrate=BiTemporalMemory()` to `skill_organism(...)`, stages can:

- **Read** facts via `read_query` — a subject string or callable returning a `BiTemporalQuery`
- **Write** facts via `emit_output_fact=True` (auto-records output) or `fact_extractor` (custom event logic)

Stages receive a frozen `SubstrateView(facts, query, record_time)` rather than the raw memory instance, keeping them decoupled from memory internals.

This enables the audit question: "what did the organism know when stage X made its decision?" — answered via `retrieve_belief_state()` on the append-only history.

### Attachable components

For cross-cutting concerns you do not want to hardcode:

- telemetry (`TelemetryProbe`)
- runtime monitoring (`WatcherComponent`) — classifies signals as epistemic/somatic/species-specific, can retry/escalate/halt
- review and safety policies
- custom lifecycle hooks via `SkillRuntimeComponent`

## Recommended Examples

- [`examples/68_skill_organism_runtime.py`](https://github.com/coredipper/operon/blob/main/examples/68_skill_organism_runtime.py) — deterministic intake, fast routing, deep planning, attached telemetry
- [`examples/71_bitemporal_skill_organism.py`](https://github.com/coredipper/operon/blob/main/examples/71_bitemporal_skill_organism.py) — multi-stage workflow with bi-temporal substrate, belief-state reconstruction, and temporal diffs
- [`examples/73_watcher_component.py`](https://github.com/coredipper/operon/blob/main/examples/73_watcher_component.py) — runtime monitoring with signal classification and retry/escalate/halt interventions
