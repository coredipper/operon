---
title: Operon Diagram Builder
emoji: "\U0001F3D7"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Build custom wiring diagrams and get full epistemic analysis
---

# Operon Diagram Builder

Build **custom wiring diagrams** via text definitions, then get full structural and epistemic analysis including topology classification, observation profiles, and theorem predictions.

## What to Try

1. Open the **Build & Analyze** tab, select a preset or type your own module/wire definitions, then click **Build & Analyze** to see the full epistemic breakdown.
2. Switch to the **Compare Topologies** tab, define two diagrams side by side, and click **Compare** to see classification and theorem values for both.

## How It Works

Define modules as `Name:atp:latency[:cap1,cap2]` and wires as `Src.port -> Dst.port[:denature|:optic]`. The builder parses your definitions into a wiring diagram and runs the full epistemic analysis pipeline: observation profiles, partition, topology classification, and all four theorem predictions.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
