---
title: Operon Convergence Compiler
emoji: "\u2699\uFE0F"
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Compile organisms and verify certificates
---

# Operon Convergence Compiler

Compile a multi-stage Operon organism into an external agent framework (Swarms, DeerFlow, Ralph, Scion) and verify that structural certificates survive the translation.

## What to Try

1. Click **Compile** with the defaults to see a 3-stage pipeline compiled to Swarms with passing certificates.
2. Switch the **Target Framework** dropdown to DeerFlow, Ralph, or Scion to see how the same organism maps to different orchestration shapes.
3. Check **Set budget to 0** and click **Compile** to see a failing certificate -- the priority gating guarantee cannot hold when there is no energy to gate.
4. Try the "Research pipeline" or "Code review" presets for different stage configurations.

## How It Works

The convergence compiler translates Operon's `SkillOrganism` into framework-specific config dicts. Each compiled output includes **certificates** -- self-verifiable structural guarantees (e.g. priority gating) that the compiler preserves through compilation. The verifier re-derives each guarantee from its parameters to confirm it still holds.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
