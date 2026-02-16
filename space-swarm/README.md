---
title: Operon Regenerative Swarm
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Regenerative swarm with apoptosis and worker regeneration
---

# 🧬 Regenerative Swarm

Simulate a swarm of workers that detect when they're stuck (entropy collapse), self-terminate (apoptosis), and respawn with summarized learnings from predecessors.

## Features

- **5 presets**: Stuck worker, quick solver, gradual convergence, error-prone recovery, max regenerations exhausted
- **Tunable parameters**: Entropy threshold, max steps, max regenerations
- **Full instrumentation**: Worker timeline, apoptosis events, regeneration events with injected hints

## How It Works

The `RegenerativeSwarm` supervisor monitors worker entropy. When a worker's output becomes repetitive (entropy collapse), it triggers apoptosis — the worker self-terminates, its memory is summarized, and a successor is spawned with those learnings injected as hints.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
