# Operon 🧬

**Biologically inspired architectures for more reliable AI agent systems**

> *From agent heuristics toward structural guarantees.*

![Status](https://img.shields.io/badge/status-experimental-orange)
![Version](https://img.shields.io/badge/pypi-v0.32.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Publish to PyPI](https://github.com/coredipper/operon/actions/workflows/publish.yml/badge.svg)](https://github.com/coredipper/operon/actions/workflows/publish.yml)

> Operon is a research-grade library and reference implementation for biologically inspired agent control patterns. The API is still evolving.

## The Problem: Fragile Agents

Most agent systems fail structurally, not just locally.

A worker can hallucinate and nobody checks it. A sequential chain accumulates handoff cost. A tool-rich workflow becomes harder to route safely than a single-agent baseline. In practice, adding more agents often adds more failure surface unless the wiring is doing real control work.

Operon is a library for making that structure explicit. It gives you pattern-first building blocks like reviewer gates, specialist swarms, skill organisms, and topology advice, while keeping the lower-level wiring and analysis layers available when you need them.

## Installation

```bash
pip install operon-ai
```

For provider-backed stages, configure whichever model backend you want to use through the existing `Nucleus` provider layer.

## Start Here: Pattern-First API

If you are new to Operon, start here rather than with the full biological vocabulary.

- `advise_topology(...)` when you want architecture guidance
- `reviewer_gate(...)` when you want one worker plus a review bottleneck
- `specialist_swarm(...)` when you want centralized specialist decomposition
- `skill_organism(...)` when you want a provider-bound workflow with cheap vs expensive stages and attachable telemetry
- `managed_organism(...)` when you want the full stack — adaptive assembly, watcher, substrate, development, social learning — in one call

### Get topology advice

```python
from operon_ai import advise_topology

advice = advise_topology(
    task_shape="sequential",
    tool_count=2,
    subtask_count=3,
    error_tolerance=0.02,
)

print(advice.recommended_pattern)  # single_worker_with_reviewer
print(advice.suggested_api)        # reviewer_gate(...)
print(advice.rationale)
```

### Add a reviewer gate

```python
from operon_ai import reviewer_gate

gate = reviewer_gate(
    executor=lambda prompt: f"EXECUTE: {prompt}",
    reviewer=lambda prompt, candidate: "safe" in prompt.lower(),
)

result = gate.run("Deploy safe schema migration")
print(result.allowed)
print(result.output)
```

### Build a skill organism

```python
from operon_ai import MockProvider, Nucleus, SkillStage, TelemetryProbe, skill_organism

fast = Nucleus(provider=MockProvider(responses={
    "return a deterministic routing label": "EXECUTE: billing",
}))
deep = Nucleus(provider=MockProvider(responses={
    "billing": "EXECUTE: escalate to the billing review workflow",
}))

organism = skill_organism(
    stages=[
        SkillStage(name="intake", role="Normalizer", handler=lambda task: {"request": task}),
        SkillStage(
            name="router",
            role="Classifier",
            instructions="Return a deterministic routing label.",
            mode="fixed",
        ),
        SkillStage(
            name="planner",
            role="Planner",
            instructions="Use the routing result to propose the next action.",
            mode="fuzzy",
        ),
    ],
    fast_nucleus=fast,
    deep_nucleus=deep,
    components=[TelemetryProbe()],
)

result = organism.run("Customer says the refund never posted.")
print(result.final_output)
```

### Drop down a layer when you need to

The pattern layer is additive, not a separate framework. You can still inspect the generated structure and analysis underneath. For a gate returned by `reviewer_gate(...)`:

- `gate.diagram`
- `gate.analysis`

For a swarm returned by `specialist_swarm(...)`:

- `swarm.diagram`
- `swarm.analysis`

### Bi-Temporal Memory

Append-only factual memory with dual time axes (valid time vs record time) for auditable decision-making. Stages can read from and write to a shared `BiTemporalMemory` substrate, enabling belief-state reconstruction ("what did the organism know when stage X decided?").

```python
from operon_ai import BiTemporalMemory, MockProvider, Nucleus, SkillStage, skill_organism

mem = BiTemporalMemory()
nucleus = Nucleus(provider=MockProvider(responses={}))

organism = skill_organism(
    stages=[
        SkillStage(
            name="research",
            role="Researcher",
            handler=lambda task: {"risk": "medium", "sector": "fintech"},
            emit_output_fact=True,  # records output under subject=task
        ),
        SkillStage(
            name="strategist",
            role="Strategist",
            handler=lambda task, state, outputs, stage, view: f"Recommend based on {len(view.facts)} facts",
            read_query="Review account acct:1",  # must match the task string used as subject
        ),
    ],
    fast_nucleus=nucleus,
    deep_nucleus=nucleus,
    substrate=mem,
)

result = organism.run("Review account acct:1")
print(mem.history("Review account acct:1"))  # full append-only audit trail
```

See the [Bi-Temporal Memory docs](https://banu.be/operon/bitemporal-memory/), [examples 69–71](examples/), and the [interactive explorer](https://huggingface.co/spaces/coredipper/operon-bitemporal).

### Convergence: Structural Analysis for External Frameworks

The `operon_ai.convergence` package provides typed adapters for 6 external agent frameworks ([Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), [AnimaWorks](https://github.com/xuiltul/animaworks), [Ralph](https://github.com/mikeyobrien/ralph-orchestrator), [A-Evolve](https://github.com/A-EVO-Lab/a-evolve), [Scion](https://github.com/GoogleCloudPlatform/scion)) into Operon's structural analysis layer. No external dependencies — all operate on plain dicts.

```python
from operon_ai import PatternLibrary
from operon_ai.convergence import (
    parse_swarm_topology, analyze_external_topology,
    seed_library_from_swarms, get_builtin_swarms_patterns,
)

# Analyze a Swarms workflow with Operon's epistemic theorems
topology = parse_swarm_topology(
    "HierarchicalSwarm",
    agent_specs=[
        {"name": "manager", "role": "Manager"},
        {"name": "coder", "role": "Developer"},
        {"name": "reviewer", "role": "Reviewer"},
    ],
    edges=[("manager", "coder"), ("manager", "reviewer")],
)
result = analyze_external_topology(topology)
print(result.risk_score, result.warnings)

# Seed a PatternLibrary from Swarms' built-in patterns
library = PatternLibrary()
seed_library_from_swarms(library, get_builtin_swarms_patterns())
```

Compile organisms into deployment configs for [Swarms](https://github.com/kyegomez/swarms), [DeerFlow](https://github.com/bytedance/deer-flow), [Ralph](https://github.com/mikeyobrien/ralph-orchestrator), and [Scion](https://github.com/GoogleCloudPlatform/scion):

```python
from operon_ai.convergence import organism_to_swarms, organism_to_scion
swarms_config = organism_to_swarms(organism)
scion_config = organism_to_scion(organism, runtime="docker")
```

Compile to LangGraph with all structural guarantees enforced natively (requires `pip install operon-ai[langgraph]`):

```python
from operon_ai.convergence.langgraph_compiler import run_organism_langgraph

# Works with any organism — multi-stage pipelines included
result = run_organism_langgraph(organism, task="Review this code")
print(result.output, result.interventions, result.certificates_verified)
```

See [examples 86–108](examples/) and the [Convergence docs](https://banu.be/operon/convergence/).

## Learn More

Public docs now live at [banu.be/operon](https://banu.be/operon/). The tracked source for that docs shell lives in the repo under [`docs/site/`](https://github.com/coredipper/operon/tree/main/docs/site).

- [Getting Started](https://banu.be/operon/getting-started/)
- [Pattern-First API](https://banu.be/operon/pattern-first-api/)
- [Skill Organisms](https://banu.be/operon/skill-organisms/)
- [Bi-Temporal Memory](https://banu.be/operon/bitemporal-memory/)
- [Convergence](https://banu.be/operon/convergence/)
- [Examples](https://banu.be/operon/examples/)
- [Concepts and Architecture](https://banu.be/operon/concepts/)
- [Theory and Papers](https://banu.be/operon/theory/)
- [API Overview](https://banu.be/operon/api/)
- [Hugging Face Spaces](https://banu.be/operon/spaces/)
- [Release Notes](https://banu.be/operon/releases/)

Direct links:

- [Examples index](https://github.com/coredipper/operon/blob/main/examples/README.md) (116 runnable examples)
- [Wiring diagrams](https://github.com/coredipper/operon/blob/main/examples/wiring_diagrams.md) (63 architecture diagrams)
- [Main whitepaper](https://github.com/coredipper/operon/blob/main/article/main.pdf)
- [Epistemic topology paper](https://github.com/coredipper/operon/blob/main/article/paper1/main.pdf)
- [PyPI package](https://pypi.org/project/operon-ai/)
- [Epistemic Topology Explorer](https://huggingface.co/spaces/coredipper/operon-epistemic)
- [Diagram Builder](https://huggingface.co/spaces/coredipper/operon-diagram-builder)
- [Bi-Temporal Memory Explorer](https://huggingface.co/spaces/coredipper/operon-bitemporal)

## Contributing

Issues and pull requests are welcome. Start with the pattern-first examples, then drop into the lower-level layers only when the problem actually needs them.

## License

MIT
