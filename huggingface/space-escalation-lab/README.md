---
title: Operon Escalation Lab
emoji: "\U0001F9EA"
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Quality-based model escalation demo
---

# Operon Escalation Lab

Explore **quality-based escalation**: the VerifierComponent (adaptive immunity) scores each stage's output against a rubric, and the WatcherComponent (innate immunity) escalates from the fast model to the deep model when quality drops below threshold.

## What to Try

1. Click **Run** with "Shallow bug fix" -- the fast model scores 0.25 (below 0.50 threshold), triggering escalation to the deep model.
2. Try "Adequate response" -- the fast model scores 0.85 (above threshold), so no escalation occurs.
3. Adjust the **Quality Threshold** slider to see how changing the threshold affects escalation behavior.
4. Try "Vague summary" with different thresholds to find the tipping point.

## How It Works

1. **VerifierComponent** evaluates output quality via a rubric function (0.0-1.0)
2. If quality < threshold, it emits a `WatcherSignal(category=EPISTEMIC, source="verifier")`
3. **WatcherComponent** detects the low-quality signal on the fast model
4. Watcher decides to **ESCALATE** -- re-runs the stage with the deep nucleus
5. Final output comes from the deep model

## Biological Analogy

- **Innate immunity** (WatcherComponent): generic anomaly detection via baseline deviations
- **Adaptive immunity** (VerifierComponent): specific quality assessment via rubric, like B-cells producing antibodies tailored to an antigen

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
