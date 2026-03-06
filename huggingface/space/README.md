---
title: Operon Chaperone
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Recover structured data from malformed LLM output
---

# 🧬 Operon Chaperone

Recover structured data from malformed LLM output using a multi-strategy cascade -- like protein chaperones refolding misfolded proteins.

## What to Try

1. Open the **Input & Parse** tab, select a schema from the **Target Schema** dropdown (e.g. FunctionCall), paste broken JSON into the input box, and click **Parse** to watch the cascade recover it.
2. Toggle individual strategies on or off (STRICT, EXTRACTION, LENIENT, REPAIR) to see which one succeeds for different kinds of malformation.
3. Switch to the **Batch Examples** tab and click **Run All Examples** to see the cascade handle trailing commas, single quotes, markdown-wrapped JSON, and more in one pass.

## How It Works

The Chaperone tries four folding strategies in order -- strict parsing, extraction from wrappers, lenient type coercion, and structural repair -- stopping at the first success. This mirrors how biological chaperones apply progressively stronger refolding forces to recover functional protein structure.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
