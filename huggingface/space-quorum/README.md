---
title: Operon Quorum Sensing
emoji: "\U0001F5F3\uFE0F"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent voting simulator with 7 consensus strategies
---

# 🗳️ Operon Quorum Sensing

Multi-agent voting simulator with 7 consensus strategies -- like bacterial quorum sensing where cells coordinate behavior based on population-level signals.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Unanimous agreement", "Split decision", or "Confidence filtering") and click **Run Vote** to see the result and a comparison across all 7 strategies.
2. Configure the 5 agent rows manually -- set each agent's **Name**, **Weight**, **Vote** (Permit/Block/Abstain), and **Confidence** slider, then pick a **Strategy** and run the vote.
3. Switch between strategies (Majority, Supermajority, Unanimous, Weighted, Confidence, Bayesian, Threshold) to see how the same set of votes produces different outcomes.

## How It Works

QuorumSensing aggregates agent votes using the selected strategy -- from simple majority to Bayesian belief aggregation. Weights and confidence scores influence the outcome, and the strategy comparison table shows how the same votes would be decided under each of the 7 available strategies.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
