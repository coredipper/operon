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
short_description: Pattern presets, custom diagrams, topology analysis
---

# Operon Diagram Builder

Build **custom wiring diagrams** via text definitions, then get full structural and epistemic analysis including topology classification, observation profiles, and theorem predictions.

If you are evaluating Operon for practical value, start from a pattern preset like **Reviewer Gate** or **Specialist Swarm**. The point of this Space is not to force you to think in diagrams first; it is to let you inspect the structure behind a pattern once you have something concrete to compare.

## What to Try

1. Open the **Build & Analyze** tab, select **Reviewer Gate** or **Specialist Swarm**, then click **Build & Analyze** to see the full structural breakdown.
2. Edit the module or wire text if you want to tweak the pattern rather than starting from scratch.
3. Switch to the **Compare Topologies** tab to compare a practical pattern against a more naive structure.

## How It Works

Define modules as `Name:atp:latency[:cap1,cap2]` and wires as `Src.port -> Dst.port[:denature|:optic]`. The builder parses your definitions into a wiring diagram and runs the full epistemic analysis pipeline: observation profiles, partition, topology classification, and all four theorem predictions. The pattern presets are there to make the first question easier: what should I try, and why would it help?

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
