---
title: Operon Diffusion
emoji: 🌊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Morphogen gradient formation on graph topologies
---

# 🌊 Diffusion — Morphogen Gradient Visualizer

Simulate **morphogen diffusion** across graph topologies and watch concentration gradients form — the spatial coordination layer of Operon.

## Features

- **Tab 1 — Linear Chain**: Emit morphogen from a chosen node in a linear graph, run N diffusion steps, and visualize concentration bars per node
- **Tab 2 — Topologies**: Choose from Linear, Star, Ring, Grid (2×3), or Binary Tree graphs and see how topology shapes the gradient
- **Tab 3 — Competing Sources**: Place two different morphogens at different nodes and observe overlapping gradients

## How It Works

`DiffusionField` manages a graph of nodes connected by edges. `MorphogenSource` objects emit at fixed rates. Each step: (1) emit, (2) diffuse — a fraction flows to neighbors, (3) decay — concentrations degrade, (4) clamp — cap at 1.0, snap near-zero to 0. `get_local_gradient()` bridges each node's concentrations to the `MorphogenGradient` API for agent-level strategy hints.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
