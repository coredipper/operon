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

Two demos in one: **structural healing** of invalid LLM output via the Chaperone Loop, and **context pruning** via the Autophagy Daemon.

## Features

- **Tab 1 — Healing Loop**: Mock LLM generates invalid JSON → Chaperone validates → error feedback triggers refolding
- **Tab 2 — Autophagy**: Monitor context window fullness and prune when critical
- **7 presets**: Valid first try, healed after retry, degraded, complex schema healing, healthy context, critical context, force prune

## How It Works

The `ChaperoneLoop` wraps a generator + Chaperone validator. Invalid output triggers refolding attempts where error messages are fed back to the generator. The `AutophagyDaemon` monitors context fill percentage and triggers pruning when toxicity threshold is exceeded.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
