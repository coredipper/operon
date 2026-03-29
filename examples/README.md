# Operon Examples

This directory contains 96 runnable numbered examples demonstrating the
`operon_ai` library, progressing from basic concepts to complete
LLM-powered cell simulations.

## Recommended Starting Points

If you are evaluating Operon for practical value rather than formal structure,
start here:

- [`68_skill_organism_runtime.py`](68_skill_organism_runtime.py) if you want
  the clearest answer to “can this improve a real workflow?” It binds stages to
  fast vs deep models, preserves structure, and lets you add telemetry without
  rewriting the skill.
- [`67_pattern_first_api.py`](67_pattern_first_api.py) if you want the shortest
  path to topology advice, reviewer gates, and specialist swarms without
  touching the substrate directly.
- [`66_epistemic_topology.py`](66_epistemic_topology.py) if you want to inspect
  the structural analysis underneath those recommendations.

The older examples still matter, but these two are the fastest way to answer
the question most engineers actually care about: does this help me build a
better agent system?

For **temporal memory** scenarios — tracking facts that change over time and
reconstructing past belief states:
- [`69_bitemporal_memory.py`](69_bitemporal_memory.py) — core valid-time vs
  record-time divergence with corrections
- [`70_bitemporal_compliance_audit.py`](70_bitemporal_compliance_audit.py) —
  multi-fact compliance audit with belief-state reconstruction
- [`71_bitemporal_skill_organism.py`](71_bitemporal_skill_organism.py) —
  multi-stage organism with bi-temporal substrate for auditable shared facts
- [`72_pattern_repository.py`](72_pattern_repository.py) —
  register, retrieve, and score reusable collaboration pattern templates
- [`73_watcher_component.py`](73_watcher_component.py) —
  runtime monitoring with signal classification and retry/escalate/halt interventions
- [`74_adaptive_assembly.py`](74_adaptive_assembly.py) —
  task fingerprinting, template selection, automatic organism construction, and outcome recording
- [`75_experience_driven_watcher.py`](75_experience_driven_watcher.py) —
  watcher experience pool with intervention history driving future recommendations
- [`76_cognitive_modes.py`](76_cognitive_modes.py) —
  System A/B cognitive mode annotations with watcher mode balance reporting
- [`77_sleep_consolidation.py`](77_sleep_consolidation.py) —
  sleep consolidation cycle: replay, compress, counterfactual, histone promotion
- [`78_social_learning.py`](78_social_learning.py) —
  cross-organism template sharing with trust-weighted adoption (epistemic vigilance)
- [`79_curiosity_driven_exploration.py`](79_curiosity_driven_exploration.py) —
  curiosity signals triggering escalation on novel inputs
- [`80_developmental_staging.py`](80_developmental_staging.py) —
  developmental stages, capability gating, and critical periods
- [`81_critical_periods.py`](81_critical_periods.py) —
  teacher-learner scaffolding with developmental awareness
- [`82_managed_organism.py`](82_managed_organism.py) —
  one-call managed_organism() wiring the full v0.19-0.23 stack
- [`83_cli_stage_handler.py`](83_cli_stage_handler.py) —
  shell out to external CLI tools (Claude Code, Copilot, ruff) as organism stages
- [`84_cli_organism.py`](84_cli_organism.py) —
  full managed CLI organism from a dict of commands with watcher and substrate
- [`85_claude_code_pipeline.py`](85_claude_code_pipeline.py) —
  live 3-stage pipeline (plan → implement → review) using `claude --print` with context chaining
- [`86_swarms_topology_analysis.py`](86_swarms_topology_analysis.py) —
  analyze Swarms workflow patterns with Operon's epistemic theorems
- [`87_animaworks_role_mapping.py`](87_animaworks_role_mapping.py) —
  map AnimaWorks organizational configs to Operon's typed stage system
- [`88_deerflow_workflow_analysis.py`](88_deerflow_workflow_analysis.py) —
  analyze DeerFlow 2.0 session configs with Operon's epistemic theorems
- [`89_seeded_library.py`](89_seeded_library.py) —
  seed PatternLibrary from Swarms, DeerFlow, and ACG survey patterns
- [`90_hybrid_assembly.py`](90_hybrid_assembly.py) —
  hybrid assembly with library-first + LLM generator fallback
- [`91_deerflow_skill_bridge.py`](91_deerflow_skill_bridge.py) —
  bidirectional DeerFlow Markdown skill to PatternTemplate conversion
- [`92_memory_bridge.py`](92_memory_bridge.py) —
  bridge AnimaWorks and DeerFlow memory into bi-temporal facts
- [`93_priming_view.py`](93_priming_view.py) —
  multi-channel PrimingView extending SubstrateView
- [`94_heartbeat_daemon.py`](94_heartbeat_daemon.py) —
  idle-time consolidation via HeartbeatDaemon
- [`95_async_thinking.py`](95_async_thinking.py) —
  Fork/Join sub-queries within a single stage
- [`96_codesign_composition.py`](96_codesign_composition.py) —
  Zardini co-design adapter composition with fixed-point convergence

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
| 37 | Formal Theory | Metabolic Coalgebra, budget-bounded halting conditions |
| 38-41 | Healing | Budget Tracking, Chaperone Loop, Regenerative Swarm, Autophagy |
| 42-44 | Health & Coordination | Epiplexity, Innate Immunity, Morphogen Gradients |
| 45-47 | Practical Applications | Code Review, Codebase Q&A, Cost Attribution |
| 48-55 | Orchestration | Multi-motif composition, LLM integration, capstone |
| 56-63 | Advanced Biology | Epigenetic coupling, Cell types, Tissue, Plasmids, Morphogens |
| 64-65 | Optimization & Providers | Diagram optimization, OpenAI-compatible servers |
| 66 | Epistemic Topology | Classification, error bounds, parallelism, recommendations |
| 67 | Pattern-First API | Reviewer gates, specialist swarms, topology advice |
| 68 | Skill Organisms | Provider-bound stages, fast/deep routing, attachable telemetry |
| 69-71 | Temporal Memory | Bi-temporal facts, belief-state reconstruction, compliance audit |
| 72-75 | Adaptive Structure | Pattern repository, watcher, adaptive assembly, experience pool |
| 76-79 | Cognitive Architecture | Cognitive modes, sleep consolidation, social learning, curiosity |
| 80-81 | Developmental Staging | Telomere-driven stages, critical periods, teacher-learner scaffolding |
| 82 | Managed Organism | One-call `managed_organism()` wiring full v0.19–0.23 stack |
| 83-85 | CLI Integration | Shell-out stage handlers, CLI organisms, Claude Code pipelines |

## Running Examples

```bash
# Basic examples (no LLM required)
python examples/01_code_review_bot.py

# Practical entry point (no LLM required)
python examples/67_pattern_first_api.py --test

# Provider-bound skill organism (mock providers by default)
python examples/68_skill_organism_runtime.py --test

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
