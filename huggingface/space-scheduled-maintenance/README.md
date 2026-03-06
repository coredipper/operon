---
title: Operon Scheduled Maintenance
emoji: "\U0001F504"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Oscillator-scheduled autophagy with feedback control
---

# 🔄 Operon Scheduled Maintenance

Oscillator-driven autophagy that prunes noisy context on a schedule, with a feedback loop maintaining noise ratio at a target setpoint -- like biological circadian rhythms governing cellular cleanup cycles.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Gradual pollution" or "Sudden noise spike") and click **Run Simulation** to see how the oscillator schedules maintenance windows and autophagy prunes noise.
2. Adjust the **Oscillator frequency** slider to change how often maintenance runs, and the **Noise rate** slider to control how fast context gets polluted.
3. Try "Feedback correction" to see how the NegativeFeedbackLoop aggressively tightens the toxicity threshold when noise exceeds the setpoint, then converges to homeostasis.

## How It Works

A sine-based oscillator computes maintenance windows -- autophagy runs during peak amplitude phases to prune noisy context. A NegativeFeedbackLoop adjusts the toxicity threshold toward the noise setpoint: if noise is too high, the threshold tightens (more aggressive pruning); if too low, it loosens. The Lysosome disposes of extracted waste.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
