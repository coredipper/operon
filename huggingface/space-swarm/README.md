---
title: Operon Regenerative Swarm
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Regenerative swarm with apoptosis and worker regeneration
---

# 🧬 Regenerative Swarm

Workers detect entropy collapse, self-terminate via apoptosis, and respawn with summarized learnings -- like biological tissue regeneration where dying cells pass signals to their replacements.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Stuck worker (entropy collapse)" or "Error-prone recovery") and click **Run Swarm** to see workers spawn, get stuck, die, and regenerate.
2. Adjust the **Entropy threshold** slider to control how quickly stuck workers are killed -- lower values trigger apoptosis faster.
3. Try "Max regenerations exhausted" to see what happens when the swarm runs out of regeneration attempts and fails, then compare with "Stuck worker" where the successor solves it with inherited hints.

## How It Works

The RegenerativeSwarm supervisor monitors worker output entropy. When output becomes repetitive (entropy collapse), it triggers apoptosis -- the worker self-terminates, its memory is summarized, and a successor spawns with those learnings as hints. This mirrors how tissues regenerate: dying cells release signals that guide replacement cells.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
