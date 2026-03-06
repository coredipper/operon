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

# 🧪 Operon Denaturation Layers

Wire-level anti-injection filters that strip syntactic structure from data flowing between agents -- like protein denaturation destroying the tertiary structure that pathogens exploit.

## What to Try

1. Open the **Single Filter** tab, select a filter type from the dropdown (StripMarkup, Normalize, Summarize, or Chain), paste a prompt injection payload into the input, and click **Apply** to see the denatured output.
2. Try the "ChatML injection" or "Nested code block" presets from the **Example** dropdown to see how StripMarkup neutralizes common attack vectors.
3. Switch to the **Wire Data Flow** tab and run the same input through a wiring diagram to compare raw vs. denatured data as it flows between modules.

## How It Works

Denaturation filters transform data on the wire between agents, stripping markup tokens, normalizing Unicode, and truncating content. This destroys the syntactic structure that injection payloads rely on while preserving semantic meaning -- preventing prompt injection from cascading through multi-agent systems.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
