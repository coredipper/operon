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

# 🔬 Optics -- Wire Optic Router

Type-safe data routing and transformation on wires -- prisms filter by data type, traversals map over collections, and composed optics chain into pipelines.

## What to Try

1. Open the **Prism Routing** tab, configure two prism wires with different accepted DataTypes, send data through, and see which destination receives it based on type matching.
2. Switch to the **Traversal Transform** tab, enter a list of values, pick a transform (double, uppercase, square, negate), and click **Apply** to see element-wise processing.
3. In the **Composed Optics** tab, chain a prism filter with a traversal transform and test the pipeline with various DataType values to see how both layers compose.
4. Try the **Optic + Denature** tab to see denaturation filters and optic routing working together on the same wire.

## How It Works

PrismOptic accepts specific DataType values and rejects everything else, enabling type-safe fan-out routing. TraversalOptic maps transforms over collections. ComposedOptic chains optics sequentially -- all must accept, and transforms apply left-to-right. On a Wire, denaturation runs first to strip injection vectors, then the optic routes and transforms.

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
