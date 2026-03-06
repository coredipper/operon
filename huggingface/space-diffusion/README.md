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

# 🌊 Diffusion -- Morphogen Gradient Visualizer

Simulate morphogen diffusion across graph topologies and watch concentration gradients form -- the spatial coordination layer that lets distributed agents sense their position.

## What to Try

1. Open the **Linear Chain** tab, set the **Source Node** and **Diffusion Steps** sliders, and click **Run Diffusion** to watch concentration spread from the source along a line of nodes.
2. Switch to the **Topologies** tab and select different graph shapes (Star, Ring, Grid, Binary Tree) to see how network structure shapes gradient formation.
3. In the **Competing Sources** tab, place two morphogens at different nodes and run diffusion to observe overlapping gradients and competition zones.

## How It Works

A DiffusionField manages a graph where MorphogenSource objects emit signals that spread to neighbors, decay over time, and clamp at saturation. Each node's local concentration maps to the MorphogenGradient API, giving agents position-aware strategy hints -- like how cells in a developing embryo read morphogen concentrations to determine their fate.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
