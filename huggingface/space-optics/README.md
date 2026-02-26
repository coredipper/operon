---
title: Operon Optics
emoji: 🔬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Prism routing and traversal transforms on wiring diagrams
---

# 🔬 Optics — Wire Optic Router

Explore **optic-based wiring** where prisms route data by type and traversals transform collections — the type-safe routing layer of Operon.

## Features

- **Tab 1 — Prism Routing**: Configure two prism wires with different accepted DataTypes, send data through, and see which destination receives it
- **Tab 2 — Traversal Transform**: Apply element-wise transforms (double, uppercase, square, negate) to lists via TraversalOptic
- **Tab 3 — Composed Optics**: Chain a prism filter with a traversal transform and test the pipeline with various DataTypes
- **Tab 4 — Optic + Denature**: Attach both a DenatureFilter and an Optic to a wire and see both layers compose in sequence

## How It Works

`PrismOptic` accepts a set of `DataType` values and rejects everything else — enabling fan-out routing. `TraversalOptic` maps a transform over list elements. `ComposedOptic` chains optics sequentially: all must accept, transforms apply left-to-right. On a `Wire`, denaturation runs first (stripping injection vectors), then the optic routes and transforms.

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/)
