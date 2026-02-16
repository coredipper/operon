---
title: Operon Complete Cell
emoji: "\U0001F9EC"
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Full 5-organelle pipeline from input to output
---

# Operon Complete Cell

Send a request through the full cellular pipeline and see each organelle's output:

1. **Membrane** -- Filter malicious input (prompt injection, jailbreaks)
2. **Ribosome** -- Synthesize structured prompts from templates
3. **Mitochondria** -- Execute safe computation (AST-based, no code injection)
4. **Chaperone** -- Validate output against Pydantic schema
5. **Lysosome** -- Recycle failures, clean up waste

If any step fails, downstream organelles are skipped and the Lysosome captures the failure. The cell persists across requests, accumulating statistics.

No API keys required -- runs entirely locally.

[GitHub](https://github.com/coredipper/operon) | [Paper](https://github.com/coredipper/operon/tree/main/article)
