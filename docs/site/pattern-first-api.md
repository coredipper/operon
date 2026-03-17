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

This is the most practical new runtime in `v0.18`.

## Escape Hatch

These wrappers do not hide the structure permanently. You can still inspect the generated diagram and analysis when you need to:

- `gate.diagram`
- `gate.analysis`
- `swarm.diagram`
- `swarm.analysis`

The goal is not to remove the substrate. The goal is to stop making every user start there.
