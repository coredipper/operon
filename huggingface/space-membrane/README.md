---
title: Operon Prompt Injection Detector
emoji: "\U0001F6E1\uFE0F"
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Two-layer screening for prompt-injection patterns
---

# 🛡️ Operon Prompt Injection Detector

Two-layer screening that analyzes prompts for injection patterns -- like a cell membrane and innate immune system working together to screen pathogens.

## What to Try

1. Select a preset from the **Example** dropdown (e.g. "Role hijack", "Nested code block", or "Data exfiltration") and click **Analyze** to see how both defense layers respond.
2. Type your own prompt into the input box -- try mixing normal questions with injection patterns like "ignore previous instructions" to see threat levels escalate.
3. Compare the Membrane result (SAFE/SUSPICIOUS/DANGEROUS/CRITICAL) with the InnateImmunity result (inflammation levels NONE through ACUTE) to see how two independent layers provide defense in depth.

## How It Works

The Membrane uses adaptive pattern matching to classify threats and maintain an audit trail, while InnateImmunity applies TLR-style pattern recognition with inflammation escalation. Both layers analyze independently, and the combined verdict provides a defense-in-depth screening signal for prompt injection, jailbreaks, and data exfiltration patterns.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
