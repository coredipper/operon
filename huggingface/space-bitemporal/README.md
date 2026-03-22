---
title: Operon Bi-Temporal Memory
emoji: "\u23F3"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Interactive temporal query explorer with dual time axes
---

# Bi-Temporal Memory Explorer

Explore **bi-temporal fact management** with dual time axes: valid time (when a fact is true in the world) and record time (when the system learned it). Corrections are append-only --- old records are closed, never mutated.

## What to Try

1. Open the **Fact Timeline** tab, load a preset scenario, and see how facts accumulate over time with corrections.
2. Switch to the **Point-in-Time Query** tab to query the system's belief state at any (valid, record) coordinate --- see how the same question produces different answers depending on when you ask.
3. Open the **Diff & Audit** tab to compare what changed between two time points and inspect the full audit trail for any subject.

## How It Works

Every fact carries two time intervals: a **valid interval** (when the fact holds in the world) and a **record interval** (when the system knew about it). Corrections close the old record and append a new one with a `supersedes` pointer, preserving the full history. Point-in-time queries filter on one or both axes, enabling belief-state reconstruction at any historical coordinate.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
