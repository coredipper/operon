---
title: Operon Coalgebra
emoji: ⚙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Composable state machines with bisimulation checking
---

# ⚙️ Coalgebra — State Machine Explorer

Build, compose, and compare **coalgebraic state machines** — the formal backbone of agent state management in Operon.

## Features

- **Tab 1 — Step-by-Step**: Drive a counter coalgebra with manual inputs or comma-separated sequences and inspect the full transition trace
- **Tab 2 — Composition**: Run two machines in parallel (shared input) or sequentially (output feeds input) and watch both states evolve
- **Tab 3 — Bisimulation**: Check observational equivalence of two machines over an input sequence; see witness on divergence

## How It Works

A `Coalgebra[S, I, O]` defines a Mealy machine with `readout: S → O` (observation) and `update: S × I → S` (transition). `StateMachine` wraps a coalgebra with mutable state and a trace log. `ParallelCoalgebra` runs two machines on shared input; `SequentialCoalgebra` pipes the first machine's output into the second. `check_bisimulation` tests whether two machines produce identical outputs over a given input sequence.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
