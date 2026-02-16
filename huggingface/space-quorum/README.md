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

# Operon Quorum Sensing -- Voting Simulator

Configure a panel of agents and see how multi-agent consensus works across 7 voting strategies:

1. **Majority** -- Simple >50% majority
2. **Supermajority** -- >66% (two-thirds) required
3. **Unanimous** -- All must agree (zero blocks)
4. **Weighted** -- Weight-adjusted majority (weight * confidence)
5. **Confidence** -- Only votes above minimum confidence threshold count
6. **Bayesian** -- Bayesian belief aggregation from uniform prior
7. **Threshold** -- Fixed count of permits required

Run a vote and see both the result for your chosen strategy and a comparison across all strategies.

No API keys required -- runs entirely locally.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
