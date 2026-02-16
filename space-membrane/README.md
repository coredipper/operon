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
short_description: Two-layer defense against prompt injection attacks
---

# Operon Prompt Injection Detector

Two independent defense layers analyze prompts for injection attacks:

1. **Membrane** -- Adaptive immune system with pattern matching, threat levels (SAFE/SUSPICIOUS/DANGEROUS/CRITICAL), and audit trail
2. **InnateImmunity** -- TLR pattern matching with inflammation response escalation (NONE/LOW/MEDIUM/HIGH/ACUTE)

Type a prompt or select a preset example to see how both layers respond independently, with matched patterns, threat levels, and a combined verdict.

No API keys required -- runs entirely locally.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
