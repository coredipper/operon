---
title: Operon Biological Oscillators
emoji: 🔃
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Biological oscillator patterns and waveform visualization
---

# 🔃 Biological Oscillator Patterns

Compute and visualize oscillator waveforms (sine, square, sawtooth, triangle, pulse) with damping, plus explore biological oscillator phase structures.

## Features

- **Tab 1 — Waveform Explorer**: 5 waveform types with configurable frequency, amplitude, damping, and cycles
- **Tab 2 — Biological Oscillators**: Circadian (day/night), Heartbeat (BPM), and Cell Cycle (G1/S/G2/M) phase breakdowns
- **8 presets**: Sine wave, square pulse, damped sine, fast triangle, sawtooth, circadian, heartbeat, cell division

## How It Works

Waveforms are computed mathematically (matching `oscillator.py` internals) without starting background threads. Biological oscillators show phase structures that map to real-world patterns like circadian rhythms and cell division cycles.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
