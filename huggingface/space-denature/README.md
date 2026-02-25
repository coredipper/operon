---
title: Operon Denaturation Layers
emoji: "\U0001F9EA"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Wire-level anti-injection filters (anti-prion defense)
---

# Operon Denaturation Layers -- Anti-Prion Defense

Wire-level filters that transform data between agents to disrupt prompt injection cascading (Paper §5.3).

Like protein denaturation destroys tertiary structure, these filters strip the **syntactic structure** that injection payloads rely on while preserving semantic content.

**Filters:**
- **StripMarkup** -- Removes code blocks, ChatML tokens, [INST] tags, XML role tags
- **Normalize** -- Lowercase, strip control chars, Unicode NFKC normalization
- **Summarize** -- Truncation with prefix, collapses whitespace
- **Chain** -- Compose multiple filters in sequence

Type or paste a prompt to see how each filter transforms it, and compare raw vs. denatured wire data flow.

No API keys required -- runs entirely locally.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
