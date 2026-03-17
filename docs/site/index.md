# Operon Documentation

Operon is a research-grade Python library for building biologically inspired AI agent systems with explicit control structure.

This documentation tree is the source for the `/operon` site on `banu.be`. The goal is to keep the long-form docs close to the codebase while keeping the repository `README` short and usable.

## Start Here

- [Getting Started](getting-started.md)
- [Pattern-First API](pattern-first-api.md)
- [Skill Organisms](skill-organisms.md)
- [Examples](examples.md)

## Deeper Material

- [Concepts and Architecture](concepts.md)
- [Theory and Papers](theory.md)
- [API Overview](api.md)
- [Hugging Face Spaces](spaces.md)
- [Release Notes](releases.md)

## Current Positioning

The recommended entry point into Operon is the pattern-first API:

- `advise_topology(...)`
- `reviewer_gate(...)`
- `specialist_swarm(...)`
- `skill_organism(...)`

Those wrappers compile down to the same underlying wiring and analysis layers used by the papers and the more formal parts of the library.
