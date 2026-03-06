---
title: Operon Epiplexity Healing Cascade
emoji: "\U0001F4A1"
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Escalating healing when stagnation is detected
---

# 💡 Operon Epiplexity Healing Cascade

Detect epistemic stagnation and escalate through increasingly aggressive healing -- autophagy, regeneration, and abort -- like an immune system ramping up its response when initial measures fail.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Stagnant agent" or "Critical with regeneration") and click **Run Cascade** to see how stagnation triggers escalating healing interventions.
2. Adjust the **Stagnation threshold** and **Autophagy threshold** sliders to change when each healing level kicks in, then re-run to compare outcomes.
3. Try the "Healthy agent" preset to confirm that a well-performing agent passes through without triggering any healing.

## How It Works

The EpiplexityMonitor watches for declining novelty in agent output. When stagnation is detected, the cascade escalates: first autophagy prunes noisy context, then regeneration respawns the worker with clean state, and finally abort halts the pipeline if nothing else works.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
