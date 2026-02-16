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

# Operon Chaperone — Interactive Demo

Try the Chaperone's multi-strategy cascade for recovering structured data from malformed LLM output:

1. **STRICT** — Direct JSON parse, no modifications
2. **EXTRACTION** — Find JSON in markdown blocks, XML tags, or bare objects
3. **LENIENT** — Type coercion (e.g., `"42"` → `42`)
4. **REPAIR** — Fix trailing commas, single quotes, Python literals, unquoted keys

Paste broken JSON, pick a target schema, and watch the cascade recover it.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
