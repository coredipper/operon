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

# Operon Budget Simulator

Multi-currency metabolic energy management with three currencies:

- **ATP** -- Primary energy for general operations
- **GTP** -- Premium energy for specialized operations
- **NADH** -- Reserve that auto-converts to ATP when needed

Watch metabolic state transitions (FEASTING > NORMAL > CONSERVING > STARVING), NADH-to-ATP conversion, and graceful degradation under resource pressure.

Configure budgets, queue tasks with custom costs, and see exactly when and why agents exhaust their resources.

No API keys required -- runs entirely locally.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
