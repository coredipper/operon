---
title: Operon Morphogen Gradients
emoji: 🧪
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Gradient-based agent coordination without central control
---

# 🧪 Morphogen Gradients

Explore **gradient-based coordination** where agents adapt behavior based on local chemical signals — no central controller needed.

## Features

- **Tab 1 — Manual Gradient**: Set 6 morphogen values and see strategy hints, context injection, and phenotype adaptation
- **Tab 2 — Orchestrator Simulation**: Watch gradients evolve step-by-step as the orchestrator reacts to successes and failures
- **7 presets**: Easy task, crisis mode, exploration, budget crunch, smooth sailing, cascading failures, recovery arc

## How It Works

The `MorphogenGradient` holds 6 signal types (complexity, confidence, budget, error_rate, urgency, risk). The `GradientOrchestrator` adjusts these signals after each step result, producing strategy hints and phenotype parameters that shape agent behavior.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
