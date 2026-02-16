---
title: Operon Immunity Healing Router
emoji: "\U0001F6E1"
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Classify threats and route to different healing mechanisms
---

# Operon Immunity Healing Router

An API gateway pattern that classifies threats using InnateImmunity and routes to different healing mechanisms based on severity.

## Features

- **Threat classification**: InnateImmunity detects injection, abuse, and structural issues
- **Severity routing**: CLEAN -> passthrough, LOW -> chaperone repair, MEDIUM -> autophagy, HIGH -> reject
- **Healing pipeline**: ChaperoneLoop for structural repair, AutophagyDaemon for content cleanup
- **Presets**: Clean input, mild issues, moderate pollution, injection attack

## Motifs Combined

InnateImmunity + ChaperoneLoop + AutophagyDaemon + Cascade

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
