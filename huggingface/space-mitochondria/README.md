---
title: Operon Mitochondria
emoji: "\u26A1"
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Safe calculator with AST-based parsing -- no injection risk
---

# ⚡ Operon Mitochondria -- Safe Calculator

AST-based expression evaluation with zero code injection risk -- like mitochondria converting substrates into energy through controlled metabolic pathways.

## What to Try

1. Select a preset from the **Example** dropdown (e.g. "Math functions", "Boolean logic", or "JSON parsing") and click **Execute** to see the result and which metabolic pathway was used.
2. Switch between pathway tabs -- **Glycolysis** for math, **Krebs Cycle** for logic, **Beta Oxidation** for data parsing, or **Auto-detect** -- and try expressions in each.
3. Type `sqrt(sin(pi/2) * 16) + factorial(3)` to see nested function evaluation, or try an invalid expression to watch ROS (error damage) accumulate.

## How It Works

Expressions are parsed into an abstract syntax tree where only whitelisted operations can execute -- no `eval()`, no code injection. Four metabolic pathways handle different expression types, and ROS tracking accumulates error damage that mitophagy can repair, mirroring how mitochondria manage oxidative stress.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
