---
title: Operon Budget Simulator
emoji: "\U0001F50B"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Multi-currency metabolic energy management simulator
---

# 🔋 Operon Budget Simulator

Simulate multi-currency metabolic energy management where agents consume ATP, GTP, and NADH to execute tasks -- like cellular metabolism powering biological processes.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "NADH reserve rescue" or "Multi-currency") and click **Run Simulation** to see how tasks drain each currency.
2. Adjust the **ATP**, **GTP**, and **NADH** budget sliders, then edit the **Task queue** to add custom tasks with specific costs and watch metabolic state transitions (NORMAL, CONSERVING, STARVING).
3. Try the "Constrained agent" preset to see how the system gracefully degrades when budget runs out mid-queue.

## How It Works

The ATP_Store tracks three energy currencies with automatic NADH-to-ATP conversion when primary reserves deplete. As resources drop, the agent transitions through metabolic states that restrict which operations are allowed -- mirroring how cells shift from growth to conservation under energy stress.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
