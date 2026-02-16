---
title: Operon Morphogen-Guided Swarm
emoji: "\U0001F9EC"
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Swarm workers adapt strategy via morphogen gradients
---

# Operon Morphogen-Guided Swarm

A task-solving swarm where workers adapt their strategy based on morphogen gradient signals. Failed workers update gradients, and successors read those signals to adjust.

## Features

- **Gradient-aware workers**: Read morphogen signals to choose strategy
- **Adaptive orchestration**: GradientOrchestrator updates gradients after each step
- **Budget tracking**: ATP budget consumption visible in gradient evolution
- **Presets**: Normal solving, gradient adaptation, budget exhaustion

## Motifs Combined

MorphogenGradient + GradientOrchestrator + RegenerativeSwarm

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
