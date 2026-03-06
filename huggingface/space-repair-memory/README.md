---
title: Operon Repair Memory Agent
emoji: "\U0001F9E0"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: LLM agent that remembers which repair strategies worked
---

# 🧠 Operon Repair Memory Agent

An LLM agent that remembers which repair strategies worked and reuses them on future failures -- like epigenetic memory where cells mark successful adaptations for faster future responses.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "First failure" or "Memory reuse") and click **Run Simulation** to see how the agent validates output, detects failures, and applies repair strategies.
2. Adjust the **Temperature** slider to control output diversity and **Max repairs** to limit healing attempts, then re-run to see how these affect recovery success.
3. Try "First failure" followed by "Memory reuse" to see how the HistoneStore recalls successful repair strategies from the first run and applies them immediately in the second.

## How It Works

The Nucleus generates LLM output that the Chaperone validates against a schema. On failure, the agent consults HistoneStore for previously successful repair strategies and injects them into the ChaperoneLoop healing process. Successful repairs are stored as epigenetic markers for future reuse, while EpiplexityMonitor watches for output diversity.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
