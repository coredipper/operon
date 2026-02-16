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

# Operon Compliance Review Pipeline

A multi-stage contract/document review pipeline where cascade stages check morphogen signals and escalate to quorum voting when confidence is low.

## Features

- **Morphogen-driven review**: Complexity and risk gradients guide pipeline behavior
- **Dynamic quorum activation**: Low confidence triggers multi-agent voting
- **Feedback-adjusted confidence**: NegativeFeedbackLoop tunes thresholds
- **Presets**: Simple contract, ambiguous clauses, high-risk document

## Motifs Combined

MorphogenGradient + Cascade + QuorumSensing + NegativeFeedbackLoop

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
