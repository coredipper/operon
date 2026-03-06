---
title: Operon Lifecycle Manager
emoji: "\U0001F9EC"
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Agent telomere lifecycle and genome configuration management
---

# 🧬 Operon Lifecycle Manager

Manage agent lifespan with biological telomere shortening and genome configuration -- like chromosomal aging and gene expression controlling a cell's fate.

## What to Try

1. Open the **Telomere Lifecycle** tab, select a preset (e.g. "Fragile agent" or "Renewable agent"), set the **Operations** slider, and click **Run Lifecycle** to watch telomeres shorten and the agent transition through NASCENT, ACTIVE, and SENESCENT phases.
2. Enable the **Allow Renewal** toggle and re-run to see how telomere renewal extends the agent's lifespan beyond its normal limit.
3. Switch to the **Genome** tab, configure genes with different types (STRUCTURAL, REGULATORY, CONDITIONAL, DORMANT), click **Express** to see the active configuration, then click **Replicate** to produce a child genome with random mutations.

## How It Works

The Telomere tracks remaining operational capacity -- each operation shortens it, triggering phase transitions that restrict behavior as the agent ages. The Genome holds typed genes that can be expressed (active config) or replicated with mutations, modeling how biological cells differentiate and age.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
