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

# 🧬 Operon Morphogen-Guided Swarm

A task-solving swarm where workers read morphogen gradients to adapt strategy -- failed workers update signals so successors avoid repeating mistakes, like cells coordinating via chemical gradients.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Gradient adaptation" or "Budget exhaustion") and click **Run Swarm** to see how workers adapt strategy based on morphogen signals.
2. Adjust the **Entropy threshold** slider to control how quickly stuck workers are replaced, then compare how gradient signals evolve across worker generations.
3. Try "Budget exhaustion" to see how declining ATP budget signals force workers to switch to cheaper strategies.

## How It Works

The GradientOrchestrator updates morphogen signals (complexity, confidence, budget, error_rate) after each worker step. New workers read these gradients to choose their strategy, creating an adaptive feedback loop where the swarm learns from failures without explicit memory -- coordination emerges from the gradient field.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
