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

## Shared State vs Components

`shared_state` is still useful for lightweight orchestration data.

Attachable components are for concerns you do not want to hardcode into the workflow itself:

- telemetry
- later: review
- later: health or safety policies
- later: richer memory or substrate layers

This distinction matters because it keeps the workflow logic and the runtime concerns separable.

## Recommended Example

Start with:

- [`examples/69_skill_organism_runtime.py`](../../examples/69_skill_organism_runtime.py)

That example shows:

- deterministic intake
- fast routing
- deep planning
- attached telemetry
