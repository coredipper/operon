# Getting Started

## Installation

```bash
pip install operon-ai
```

For provider-backed workflows, configure the model backend you want to use through the `Nucleus` provider layer.

## Recommended First Steps

1. Start with the [Pattern-First API](pattern-first-api.md).
2. Run the pattern-first example and the skill-organism example.
3. Read the [Examples](examples.md) page for the shortest path into the rest of the library.

## First Practical Questions

If you are evaluating Operon, the right first questions are usually:

- should this stay single-agent?
- does it need a reviewer bottleneck?
- is it actually decomposable into specialists?
- should fixed work go to a cheap model and fuzzy work to a stronger one?

That is why the current front door is pattern-first rather than theory-first.

## Minimal Entry Points

- `advise_topology(...)` for architecture guidance
- `reviewer_gate(...)` for one-worker-plus-reviewer
- `specialist_swarm(...)` for centralized specialist decomposition
- `skill_organism(...)` for multi-stage workflows with attachable components
- `managed_organism(...)` for the full stack in one call (adaptive assembly, watcher, substrate, development, social learning). Requires either a seeded `library` with templates or explicit `stages=`.
