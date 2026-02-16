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
short_description: Dying swarm workers clean context via autophagy before passing state
---

# Operon Swarm Graceful Cleanup

LLM-powered swarm where dying workers clean up their context via autophagy before passing state to successors. Successors inherit clean summaries instead of raw noise.

## Features

- **Graceful cleanup**: AutophagyDaemon prunes context before worker death
- **Clean state transfer**: HistoneStore saves summaries for successor inheritance
- **Noise disposal**: Lysosome disposes extracted noise
- **Presets**: Research with cleanup, context pollution comparison

## Motifs Combined

Nucleus + RegenerativeSwarm + AutophagyDaemon + MorphogenGradient + HistoneStore

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
