---
title: Operon Security Lab
emoji: "\U0001F6E1"
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Prompt injection playground with layered biological defenses
---

# Operon Security Lab

Explore how Operon's layered biological defenses detect and block prompt injection attacks -- from pattern-based screening to proof-carrying certificates.

## What to Try

1. Go to the **Attack Lab** tab, select a preset attack (e.g. "Instruction Override" or "Jailbreak: Enable DAN mode"), and click **Scan** to see how each defense layer responds independently.
2. Switch to the **Layered Defense** tab, pick the same attack, and click **Run Full Pipeline** to watch it flow through all four layers: Membrane, InnateImmunity, DNA Repair, and Certificate verification.
3. Try writing your own adversarial inputs in the free-text area to test edge cases.

## How It Works

| Layer | Biological Analog | What It Does |
|-------|------------------|--------------|
| **Membrane** | Cell membrane / innate immunity | Pattern-based screening against known attack signatures (instruction overrides, jailbreaks, structural injections) |
| **InnateImmunity** | Toll-Like Receptors (TLRs) | Regex-based PAMP detection with inflammation response escalation (NONE through ACUTE) |
| **DNA Repair** | DNA damage response (DDR) | Genome state integrity checking -- detects drift from checkpointed configuration |
| **Certificate** | Proof-carrying code | Formally verifiable structural guarantee that state matches checkpoint |

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
