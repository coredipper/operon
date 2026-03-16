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
short_description: Wire-level denaturation filters for defense in depth
---

# 🧪 Operon Denaturation Layers

Wire-level denaturation filters that strip syntactic structure from data flowing between agents -- like protein denaturation disrupting the tertiary structure that pathogens exploit.

## What to Try

1. Open the **Single Filter** tab, select a filter type from the dropdown (StripMarkup, Normalize, Summarize, or Chain), paste a prompt injection payload into the input, and click **Apply** to see the denatured output.
2. Try the "ChatML injection" or "Nested code block" presets from the **Example** dropdown to see how StripMarkup neutralizes common attack vectors.
3. Switch to the **Wire Data Flow** tab and run the same input through a wiring diagram to compare raw vs. denatured data as it flows between modules.

## How It Works

Denaturation filters transform data on the wire between agents, stripping markup tokens, normalizing Unicode, and truncating content. This targets known syntactic patterns that many injection payloads rely on while preserving semantic meaning. It is a defense-in-depth layer that reduces the chance of injection cascading through multi-agent systems, not a complete security guarantee.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
