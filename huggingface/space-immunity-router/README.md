---
title: Operon Immunity Healing Router
emoji: "\U0001F6E1"
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Classify threats and route to different healing mechanisms
---

# 🛡 Operon Immunity Healing Router

Classify threats via InnateImmunity and route to escalating healing mechanisms -- passthrough, chaperone repair, autophagy cleanup, or hard reject -- like an immune system triaging pathogens by severity.

## What to Try

1. Select a preset from the **Preset** dropdown (e.g. "Clean input", "Mild issues", or "Injection attack") and click **Run Router** to see how the immunity layer classifies the threat and routes it to the appropriate healing mechanism.
2. Type your own text into the **Input** textbox -- try mixing legitimate questions with injection attempts to see how the router heals instead of rejecting.
3. Compare the "Moderate pollution" preset (routed to autophagy) vs. "Injection attack" (hard reject) to see how severity determines the response.

## How It Works

InnateImmunity scans input for injection patterns and abuse, assigning an inflammation level. Clean input passes through; low-severity issues get chaperone structural repair; medium-severity triggers autophagy cleanup; high-severity is rejected outright -- healing what can be saved while blocking genuine threats.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
