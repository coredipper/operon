---
title: Operon Plasmid Registry
emoji: "\U0001F9EC"
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Dynamic tool acquisition with capability gating (HGT)
---

# 🧬 Operon Plasmid Registry

Dynamic tool acquisition with capability gating -- like bacteria exchanging plasmids via horizontal gene transfer, where capability restrictions prevent privilege escalation.

## What to Try

1. Open the **Interactive** tab, select "READ_FS only" from the **Allowed Capabilities** dropdown, click **Create Agent**, then try acquiring `reverse` (succeeds -- no caps needed) and `fetch_url` (blocked -- requires NET capability).
2. Search the registry by typing tags like "text" or "network" in the **Search plasmids** box to browse available tools with their required capabilities.
3. Acquire a tool, type an expression like `reverse("hello world")` in the **Expression** box, and click **Execute**. Then click **Release** to remove it (plasmid curing).
4. Switch to the **Guided Scenario** tab and click **Run Scenario** for a complete walkthrough of acquire, block, execute, release, and re-acquire.

## How It Works

Agents discover tools in a PlasmidRegistry and absorb them into their Mitochondria runtime, but only if they hold the required capabilities. This prevents privilege escalation -- an agent with only READ_FS cannot acquire a tool that requires NET access, just as bacteria can only incorporate plasmids compatible with their cellular machinery.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
