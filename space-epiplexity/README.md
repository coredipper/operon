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

Feed a sequence of messages to the **EpiplexityMonitor** and watch how embedding novelty and perplexity combine to detect when an agent gets stuck repeating itself.

## Features

- **5 presets**: Healthy exploration, gradual stagnation, sudden loop, convergence, recovery
- **Tunable parameters**: Alpha mixing, window size, threshold
- **Real-time status**: HEALTHY → EXPLORING → CONVERGING → STAGNANT → CRITICAL

## How It Works

Each message is embedded and compared to the running window. **Epiplexity** = α × embedding_novelty + (1-α) × normalized_perplexity. When the integral of epiplexity drops below the threshold for `critical_duration` steps, the monitor flags stagnation.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
