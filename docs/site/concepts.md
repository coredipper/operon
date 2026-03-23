# Concepts and Architecture

Operon is built around the idea that agent reliability depends heavily on structure.

## Core View

The library borrows from biology not as decoration, but as a source of reusable control motifs:

- review bottlenecks
- consensus gates
- bounded resource use
- self-repair and surveillance
- explicit interfaces between stages

## Current Practical Layers

### Pattern-first API

The recommended front door for engineers:

- topology advice
- reviewer gates
- specialist swarms
- skill organisms

### Organelles

Shared infrastructural components such as:

- `Nucleus`
- `Membrane`
- `Mitochondria`
- `Chaperone`
- `Lysosome`

### State and Coordination

Includes:

- ATP / metabolic budgeting
- histone-based memory
- surveillance and healing
- morphogen coordination

### Typed Wiring and Analysis

The lower-level layer models modules, ports, wires, and topology explicitly so the library can reason about architecture instead of treating everything as an opaque chain.

## What Changed Recently

- `v0.17` made the epistemic topology story explicit
- `v0.18` made the practical front door thinner
- `v0.19` added bi-temporal memory with dual time axes
- `v0.20` integrated bi-temporal memory into the SkillOrganism runtime as an auditable substrate, formalizing the three-layer context model (topology, ephemeral, bi-temporal)
- `v0.21` added pattern repository, watcher component, adaptive assembly, and experience pool — the adaptive structure layer
- `v0.22` added cognitive mode annotations (System A/B) and sleep consolidation — the cognitive architecture layer

The current center of gravity is pattern-first workflows with auditable state.
