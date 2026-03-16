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
short_description: Composable state machines with bounded equivalence checks
---

# ⚙️ Coalgebra -- State Machine Explorer

Build, compose, and compare coalgebraic state machines -- the formal backbone of agent state management in Operon.

## What to Try

1. Open the **Step-by-Step** tab, pick a machine type (e.g. "Counter" or "Modular mod 10"), enter a comma-separated input sequence like `1,2,3,4,5`, and click **Run Sequence** to see the full transition trace.
2. Switch to the **Composition** tab, select two different machine types and a mode (Parallel or Sequential), then run inputs to watch both machines evolve together.
3. In the **Finite-Trace Equivalence** tab, pick two machines, enter an input sequence, and click **Check Equivalence** to see if they produce identical outputs or find the divergence witness.

## How It Works

A coalgebra defines a state machine with observation (`readout`) and transition (`update`) functions. Machines can be composed in parallel (shared input) or sequentially (output feeds input), and `check_bisimulation()` compares whether two machines are observationally equivalent over a supplied input sequence. In other words, this demo shows a bounded finite-trace equivalence check rather than an all-input proof.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
