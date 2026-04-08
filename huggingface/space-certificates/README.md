---
title: Operon Certificate Framework
emoji: "\U0001F50F"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Self-verifiable structural guarantees with tamper detection
---

# Operon Certificate Framework

Issue, verify, and tamper-detect **categorical certificates** -- self-verifiable structural guarantees from Operon components (ATP_Store, QuorumSensingBio, MTORScaler).

## What to Try

1. **Issue & Verify** -- Select a component, adjust its parameters, and click **Certify & Verify** to see the certificate theorem, conclusion, and derivation evidence. Try the "ATP empty (fails)" preset to see a failing certificate.
2. **Tamper Detection** -- Issue a certificate with healthy parameters, then use the tamper slider to inject a bad value (e.g., budget=0 for ATP). Click **Re-verify** to see the certificate honestly detect the change.

## How It Works

Certificates use *derivation replay*: `verify()` re-derives the structural guarantee from parameters rather than trusting a stored boolean. If conditions change after issuance, the certificate detects it. Three guarantees are supported:

- **priority_gating** (ATP_Store) -- budget > 0 with ordered priority thresholds
- **no_false_activation** (QuorumSensingBio) -- normal traffic stays below activation threshold
- **no_oscillation** (MTORScaler) -- hysteresis dead bands prevent state oscillation

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
