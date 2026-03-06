---
title: Operon Chaperone Healing & Autophagy
emoji: 🩹
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Chaperone healing loop and autophagy context pruning
---

# 🩹 Chaperone Healing Loop & Autophagy

Two self-repair mechanisms in one demo: structural healing of invalid LLM output and context pruning to prevent memory bloat -- like protein refolding and cellular autophagy.

## What to Try

1. Open the **Healing Loop** tab, select a preset (e.g. "Healed after retry" or "Complex schema"), and click **Run Healing** to watch the Chaperone validate output, detect errors, and refold until it produces valid JSON.
2. Switch to the **Autophagy** tab, select "Critical context" or "Force prune", and click **Run Autophagy** to see the daemon detect context pollution and prune noisy tokens.
3. Try the "Degraded (unfixable)" preset in the Healing tab to see what happens when all refolding attempts fail.

## How It Works

The ChaperoneLoop wraps an LLM generator with a Chaperone validator -- invalid output triggers refolding where error messages guide the next attempt. The AutophagyDaemon monitors context fill percentage and triggers pruning when toxicity exceeds a threshold, recycling waste through the Lysosome.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
