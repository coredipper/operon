---
title: Operon Repair Memory Agent
emoji: "\U0001F9E0"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: LLM agent that remembers which repair strategies worked
---

# Operon Repair Memory Agent

An LLM agent that remembers which repair strategies worked and reuses them on future failures, creating epigenetic repair memory.

## Features

- **Schema validation**: Chaperone validates LLM output against Pydantic schemas
- **Epigenetic memory**: HistoneStore records successful repair strategies
- **Repair reuse**: Recalled strategies injected into ChaperoneLoop healing
- **Presets**: First failure, memory reuse, diverse outputs

## Motifs Combined

Nucleus + HistoneStore + ChaperoneLoop + EpiplexityMonitor

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
