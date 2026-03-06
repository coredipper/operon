---
title: Operon Swarm Graceful Cleanup
emoji: "\U0001F9F9"
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Workers clean context via autophagy before death
---

# 🧹 Operon Swarm Graceful Cleanup

Dying workers clean up context via autophagy before passing state to successors -- inheriting clean summaries instead of raw noise, like apoptotic cells packaging their contents for recycling.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Research with cleanup" or "Context pollution comparison") and click **Run Swarm** to see workers accumulate context, clean up before death, and pass clean summaries to successors.
2. Adjust the **Entropy threshold** and **Max steps per worker** sliders to control how quickly workers get stuck and how many steps they take before cleanup.
3. Try "Multi-generation" to see how multiple generations of workers build a rich HistoneStore of clean summaries, each inheriting from the last.

## How It Works

Before a worker dies, the AutophagyDaemon prunes noisy context, the Lysosome disposes of waste, and the HistoneStore saves a clean summary. Successor workers inherit these summaries instead of raw accumulated noise, solving tasks faster because they start from distilled knowledge rather than polluted context.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
