# Operon Examples

This directory contains 45 examples demonstrating the operon_ai library,
progressing from basic concepts to complete LLM-powered cell simulations.

## Import Style Guide

All examples should follow this import pattern:

```python
# Standard library imports first
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
from pydantic import BaseModel

# Operon imports - grouped with parentheses for multiple imports
from operon_ai import (
    ATP_Store,
    Membrane,
    Signal,
    ThreatLevel,
)

# Operon submodule imports - separate import statements
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider, ProviderConfig
```

## Example Progression

| Range | Focus | Key Concepts |
|-------|-------|--------------|
| 01-07 | Basics | Topologies, Budget, Membrane |
| 08-11 | Organelles | Mitochondria, Chaperone, Ribosome, Lysosome |
| 12-16 | Integration | Complete Cell, Metabolism, Memory, Lifecycle |
| 17-18 | Advanced | WAgent, Cell Integrity |
| 19-25 | LLM Integration | Real providers, Memory, Tools |
| 26-36 | Wiring Diagrams | Visual architecture, Composition |
| 37 | Formal Theory | Metabolic Coalgebra, Halting Guarantee |
| 38-39 | Healing | Budget Tracking, Chaperone Loop |
| 40-42 | Health & Coordination | Epiplexity, Innate Immunity, Morphogen Gradients |
| 43-45 | Practical Applications | Code Review, Codebase Q&A, Cost Attribution |

## Running Examples

```bash
# Basic examples (no LLM required)
python examples/01_code_review_bot.py

# LLM examples (requires API key)
ANTHROPIC_API_KEY=sk-... python examples/19_llm_code_assistant.py --demo
```

## Naming Conventions

### Method Names

| Purpose | Preferred Name | Alternatives (avoid) |
|---------|---------------|---------------------|
| Energy consumption | `consume()` | metabolize, drain |
| Signal processing | `process()` | handle, execute |
| Template creation | `create_template()` | add_template, register_template |
| Output validation | `fold()` | validate, parse |

### Callback Names

Use `on_<event>` pattern:

- `on_state_change` - state transitions
- `on_error` - error events
- `on_complete` - completion events

### Variable Names

- `result` - operation outcomes
- `response` - LLM responses
- `signal` - input signals
- `vitals` - health status
