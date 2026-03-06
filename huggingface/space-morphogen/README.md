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

Explore gradient-based agent coordination where six chemical signals guide behavior without a central controller -- like morphogen gradients directing cell differentiation in developing embryos.

## What to Try

1. Open the **Manual Gradient** tab, adjust the six morphogen sliders (complexity, confidence, budget, error_rate, urgency, risk), and click **Analyze** to see strategy hints and phenotype adaptation.
2. Switch to the **Orchestrator Simulation** tab, pick a preset (e.g. "Crisis mode" or "Cascading failures"), and click **Run Simulation** to watch gradients evolve step-by-step as the orchestrator reacts.
3. Try "Budget crunch" to see how low budget signals change the agent's strategy, then compare with "Easy task" where all signals are favorable.

## How It Works

The MorphogenGradient holds six signal types that the GradientOrchestrator adjusts after each step based on outcomes. These signals produce strategy hints and phenotype parameters that shape agent behavior -- enabling decentralized coordination without explicit orchestration rules.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
