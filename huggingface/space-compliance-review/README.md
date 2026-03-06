---
title: Operon Compliance Review Pipeline
emoji: "\U0001F4DC"
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Document review with quorum voting on low confidence
---

# 📜 Operon Compliance Review Pipeline

Multi-stage document review where morphogen gradients guide pipeline behavior and low confidence triggers quorum voting -- like an immune system escalating its response based on threat signals.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "High-risk document" or "Ambiguous clauses") and click **Run Review** to see how the pipeline classifies and routes the document.
2. Adjust the **Complexity** and **Risk** sliders to push the morphogen gradient into different regimes, then re-run to see how the review stages change their behavior.
3. Lower the **Confidence** slider to trigger quorum voting, where multiple agents vote on ambiguous clauses using the selected voting strategy.

## How It Works

A cascade of review stages reads morphogen gradient signals (complexity, risk, confidence) to decide how thoroughly to analyze each clause. When confidence drops below a threshold, QuorumSensing activates multi-agent voting, and a NegativeFeedbackLoop adjusts confidence based on outcomes.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
