---
title: Operon LangGraph Visualizer
emoji: "\U0001F4CA"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Visualize per-stage LangGraph compilation
---

# Operon LangGraph Visualizer

Compile an organism to a **per-stage LangGraph** and visualize the graph topology. Each organism stage becomes a LangGraph node with conditional edges that route based on continue/halt decisions.

## What to Try

1. Click **Visualize & Run** with defaults to see a 3-stage graph with execution results.
2. Try the "4-stage incident" preset for a longer pipeline.
3. Try the "5-stage deep" preset to see how deep-mode stages appear in the graph.
4. Edit stages directly (name, role, mode -- one per line) to build custom topologies.
5. Uncheck "Execute after compiling" to see the topology without running.

## How It Works

`organism_to_langgraph()` creates one LangGraph node per `SkillStage`. Each node calls `organism.run_single_stage()`, so all structural guarantees (certificates, watcher interventions, halt-on-block) are handled by the organism. LangGraph provides the execution host, graph topology, observability, and checkpointing.

## Graph Topology

- **START** -> stage_1 -> stage_2 -> ... -> stage_N -> **END**
- Each stage has a conditional edge: `continue` -> next stage, `halt`/`blocked` -> END
- Stage colors indicate mode: blue = fixed, amber = fuzzy, purple = deep

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
