# Operon Examples

This directory contains 36 examples demonstrating the operon_ai library,
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

## Running Examples

```bash
# Basic examples (no LLM required)
python examples/01_code_review_bot.py

# LLM examples (requires API key)
ANTHROPIC_API_KEY=sk-... python examples/19_llm_code_assistant.py --demo
```
