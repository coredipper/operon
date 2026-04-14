---
title: Operon Harness Inspector
emoji: "\U0001F50D"
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
license: mit
short_description: Explore the Architecture triple (G, Know, Phi)
---

# Operon Harness Inspector

Explore the **Architecture triple (G, Know, Phi)** from Paper 5: *Harness Engineering as Categorical Architecture*. Build an organism, extract its categorical structure, and see how compiler functors preserve properties.

## What to Try

1. Click **Inspect** with defaults to see a 3-stage pipeline's Architecture triple.
2. Change **Stage modes** (fixed/fuzzy/deep) to see how the Phi (interface) mapping changes.
3. Set **ATP Budget to 0** to see a failing certificate in the Know component.
4. Try different presets to see how the four-pillar mapping and functor preservation vary.

## Architecture Triple

- **G (Graph)**: Syntactic wiring -- stage names and directed edges
- **Know (Knowledge)**: Structural guarantees -- self-verifiable certificates
- **Phi (Interface)**: Profunctor interface -- mode-to-model tier mapping

## References

- de los Riscos et al. [arXiv:2603.28906](https://arxiv.org/abs/2603.28906) -- ArchAgents category
- Liu [arXiv:2604.11767](https://arxiv.org/abs/2604.11767) -- Typed lambda calculus for agent composition

## Learn More

[GitHub](https://github.com/coredipper/operon) | [PyPI](https://pypi.org/project/operon-ai/) | [Paper](https://github.com/coredipper/operon/tree/main/article)
