# Pattern-First API

The pattern-first layer is the recommended public API for most users.

It gives you practical entry points while preserving the structure underneath.

## `advise_topology(...)`

Use this when you want a recommendation from task shape and operating constraints.

Typical inputs:

- task shape
- tool count
- subtask count
- error tolerance

Typical outputs:

- recommended pattern
- suggested wrapper API
- rationale

## `reviewer_gate(...)`

Use this when one worker should act, but only through an explicit review bottleneck.

This is the shortest path into the “one executor plus one reviewer” pattern.

Good fit:

- risky actions
- migrations
- approvals
- structured validation

## `specialist_swarm(...)`

Use this when the work is genuinely decomposable into specialist roles and then aggregated centrally.

Good fit:

- expert panel style workflows
- multi-perspective review
- vendor / policy / security / finance assessment

## `skill_organism(...)`

Use this when you want one coherent workflow made of multiple stages:

- deterministic handlers
- cheap / fast provider-backed stages
- deeper reasoning stages
- attachable runtime components like telemetry

Since `v0.20`, skill organisms also support an optional `BiTemporalMemory` substrate for auditable shared facts across stages. See [Skill Organisms](skill-organisms.md) for details.

## `managed_organism(...)`

Use this when you want the full v0.19–0.23 stack in one call: adaptive assembly from a pattern library, watcher with signal classification, bi-temporal substrate, developmental staging, and social learning — all opt-in via constructor parameters.

```python
from operon_ai import managed_organism, PatternLibrary, Telomere, BiTemporalMemory

m = managed_organism(
    task="Process quarterly report",
    library=lib,
    fast_nucleus=fast,
    deep_nucleus=deep,
    handlers={"intake": intake_fn, "process": process_fn},
    substrate=BiTemporalMemory(),
    telomere=Telomere(max_operations=100),
    organism_id="org-A",
)

result = m.run("Process quarterly report")
m.consolidate()
m.export_templates()
```

Methods: `.run()`, `.consolidate()`, `.export_templates()`, `.import_from_peer()`, `.scaffold()`, `.status()`. Each returns `None` if the relevant subsystem was not configured.

## `consolidate(...)`

One-call sleep consolidation. Pass a `PatternLibrary`, get back a `ConsolidationResult`:

```python
from operon_ai import consolidate
result = consolidate(library)
```

## Escape Hatch

These wrappers do not hide the structure permanently. You can still inspect the generated diagram and analysis when you need to:

- `gate.diagram`
- `gate.analysis`
- `swarm.diagram`
- `swarm.analysis`

The goal is not to remove the substrate. The goal is to stop making every user start there.
