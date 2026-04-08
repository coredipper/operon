---
title: Operon Security Audit
emoji: "\U0001F4CB"
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Security audit and certificate dashboard
---

# Operon Security Audit

Unified security audit dashboard combining DNA Repair integrity scans with categorical certificate verification across all Operon subsystems.

## What to Try

1. **Audit Pipeline** -- Select a preset agent configuration (healthy, low-budget, corrupted, silenced) and click **Run Audit** to see DNA Repair scan results, certificate status, and an overall health score.
2. **Certificate Dashboard** -- Adjust the ATP budget slider and click **Verify All** to collect and verify all four certificate types (ATP priority gating, QuorumSensing no-false-activation, MTOR no-oscillation, DNARepair state integrity). Each row shows pass/fail with evidence.

## How It Works

The audit pipeline runs a full DNA Repair scan against a genome checkpoint, collecting corruption reports by type and severity. It then issues categorical certificates from each subsystem and verifies them structurally. The overall health score is computed from both scan results and certificate status.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
