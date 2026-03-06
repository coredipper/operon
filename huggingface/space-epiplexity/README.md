---
title: Operon Epistemic Stagnation Monitor
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Epistemic stagnation detection via Bayesian surprise
---

# 🧠 Epistemic Stagnation Monitor

Detect when an agent gets stuck repeating itself by combining embedding novelty and perplexity into a single "epiplexity" score -- like neural habituation signaling that a brain region has stopped learning.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Gradual stagnation" or "Sudden loop") and click **Run Monitor** to see how the epiplexity score evolves and health status transitions from HEALTHY through STAGNANT to CRITICAL.
2. Adjust the **Alpha** slider to change the mix between embedding novelty and perplexity, then re-run to see how it affects detection sensitivity.
3. Try the "Recovery" preset to see how the monitor detects stagnation and then recovers when novel messages resume.

## How It Works

The EpiplexityMonitor embeds each message and compares it to a sliding window. Epiplexity combines embedding novelty with normalized perplexity -- when this score stays below the threshold for too long, the monitor flags stagnation, triggering healing interventions.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
