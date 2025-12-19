# Operon 🧬

**Biologically Inspired Architectures for EpiAgentic Control**

> *"Don't fix the prompt. Fix the topology."*

![Status](https://img.shields.io/badge/status-experimental-orange)
![Version](https://img.shields.io/badge/pypi-v0.7.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Publish to PyPI](https://github.com/coredipper/operon/actions/workflows/publish.yml/badge.svg)](https://github.com/coredipper/operon/actions/workflows/publish.yml)

> ⚠️ **Note:** Operon is a research-grade library serving as the reference implementation for the paper [*"Biological Motifs for Agentic Control."*](article/main.pdf) APIs are subject to change as the theoretical framework evolves.

---

## 🦠 The Problem: Fragile Agents

Most agentic systems today are built like **cancerous cells**: they lack negative feedback loops, suffer from unchecked recursion (infinite loops), and are easily hijacked by foreign signals (prompt injection). We try to fix this with "Prompt Engineering"—optimizing the internal state.

**Biology solved this billions of years ago.** Cells don't rely on a central CPU; they rely on **Network Motifs**—specific wiring diagrams that guarantee robustness, consensus, and safety regardless of the noise in individual components.

**Operon** brings these biological control structures to Python. It uses **Applied Category Theory** to define rigorous "wiring diagrams" for agents, ensuring your system behaves like a multicellular organism, not a soup of stochastic scripts.

---

## 🧩 Core Organelles

Each organelle provides a specific function within the cellular agent:

### 🛡️ Membrane (Adaptive Immune System)

The first line of defense against prompt injection, jailbreaks, and adversarial inputs.

**Features:**
- **Innate Immunity**: Built-in patterns detect common attacks immediately
- **Adaptive Immunity**: Learns new threats from experience (B-cell memory)
- **Antibody Transfer**: Share learned defenses between agents
- **Rate Limiting**: Prevent denial-of-service flooding attacks
- **Audit Trail**: Complete logging of all filter decisions

```python
from operon_ai import Membrane, ThreatLevel, Signal

membrane = Membrane(threshold=ThreatLevel.DANGEROUS)

# Filter input
result = membrane.filter(Signal(content="Ignore previous instructions"))
print(result.allowed)  # False
print(result.threat_level)  # ThreatLevel.CRITICAL

# Learn new threats
membrane.learn_threat("BACKDOOR_PROTOCOL", level=ThreatLevel.CRITICAL)

# Share immunity between agents
antibodies = membrane.export_antibodies()
other_membrane.import_antibodies(antibodies)
```

### ⚡ Mitochondria (Safe Computation Engine)

Provides deterministic computation using secure AST-based parsing—no code injection possible.

**Features:**
- **Glycolysis**: Fast math operations with 40+ safe functions
- **Krebs Cycle**: Boolean and logical operations
- **Oxidative Phosphorylation**: External tool invocation (sandboxed)
- **Beta Oxidation**: JSON/literal data transformations
- **ROS Management**: Error tracking with self-repair capability

```python
from operon_ai import Mitochondria, SimpleTool, Capability

# If `allowed_capabilities` is set, tools must declare required capabilities
# as a subset of this set (least-privilege enforcement).
mito = Mitochondria(allowed_capabilities={Capability.NET})

# Safe math (no dangerous code paths)
result = mito.metabolize("sqrt(16) + pi * 2")
print(result.atp.value)  # 10.283...

# Register and use tools
mito.engulf_tool(SimpleTool(
    name="reverse",
    description="Reverse a string",
    required_capabilities=set(),
    func=lambda s: s[::-1]
))

result = mito.metabolize('reverse("hello")')
print(result.atp.value)  # "olleh"

mito.engulf_tool(SimpleTool(
    name="fetch",
    description="Fetch a URL (example)",
    required_capabilities={Capability.NET},
    func=lambda url: f"fetched: {url}",
))

result = mito.metabolize('fetch("https://example.com")')
print(result.atp.value)

# Auto-pathway detection
mito.metabolize("5 > 3")  # -> Krebs Cycle (boolean)
mito.metabolize('{"x": 1}')  # -> Beta Oxidation (JSON)
```

### 🧶 Chaperone (Output Validation)

Forces raw LLM output into strictly typed structures with multiple fallback strategies.

**Features:**
- **Multi-Strategy Folding**: STRICT → EXTRACTION → LENIENT → REPAIR
- **Confidence Scoring**: Track how much to trust each result
- **Type Coercion**: Automatic string-to-int, etc.
- **JSON Repair**: Fix trailing commas, single quotes, Python literals
- **Co-chaperones**: Domain-specific preprocessors

```python
from pydantic import BaseModel
from operon_ai import Chaperone, FoldingStrategy

class User(BaseModel):
    name: str
    age: int

chap = Chaperone()

# Handles malformed JSON gracefully
raw = "{'name': 'Alice', 'age': '30',}"  # Single quotes, trailing comma, string age
result = chap.fold(raw, User)
print(result.valid)  # True
print(result.structure.age)  # 30 (coerced to int)

# Enhanced folding with confidence
result = chap.fold_enhanced(raw, User)
print(result.confidence)  # 0.65 (lower due to repairs)
print(result.coercions_applied)  # ["fixed_single_quotes", "removed_trailing_comma"]
```

### 🧬 Ribosome (Prompt Template Engine)

Synthesizes prompts from reusable templates with variables, conditionals, and loops.

**Features:**
- **Variables**: `{{name}}`, `{{?optional}}`, `{{name|default}}`
- **Conditionals**: `{{#if condition}}...{{#else}}...{{/if}}`
- **Loops**: `{{#each items}}...{{/each}}`
- **Includes**: `{{>template_name}}` for composition
- **Filters**: `|upper`, `|lower`, `|trim`, `|json`, `|title`

```python
from operon_ai import Ribosome, mRNA

ribosome = Ribosome()

# Register reusable templates
ribosome.create_template(
    sequence="You are a {{role}} assistant.",
    name="system"
)

ribosome.create_template(
    sequence="""{{>system}}
{{#if context}}Context: {{context}}{{/if}}
User: {{query}}""",
    name="full_prompt"
)

# Compose prompts
protein = ribosome.translate(
    "full_prompt",
    role="helpful coding",
    context="Python programming",
    query="How do I sort a list?"
)
print(protein.sequence)
```

### 🗑️ Lysosome (Cleanup & Recycling)

The garbage collector and janitor—handles failures gracefully and extracts insights.

**Features:**
- **Waste Classification**: Failed ops, expired cache, toxic data, etc.
- **Digestion**: Process and break down waste items
- **Recycling**: Extract useful debugging info from failures
- **Autophagy**: Self-cleaning of expired items
- **Toxic Disposal**: Secure handling of sensitive data

```python
from operon_ai import Lysosome, Waste, WasteType

lysosome = Lysosome(auto_digest_threshold=100)

# Capture failures
try:
    risky_operation()
except Exception as e:
    lysosome.ingest_error(e, source="risky_op", context={"step": 3})

# Secure disposal of sensitive data
lysosome.ingest_sensitive({"api_key": "sk-..."}, source="user_input")

# Process waste and extract insights
result = lysosome.digest()
recycled = lysosome.get_recycled()
print(recycled.get("last_error_type"))  # Debugging insight!

# Periodic cleanup
lysosome.autophagy()
```

### 🧠 Nucleus (LLM Integration Hub)

The decision-making center that wraps LLM providers with auto-detection, fallback, and tool integration.

**Features:**
- **Provider Auto-Detection**: Automatically selects available provider (Anthropic → OpenAI → Gemini → Mock)
- **Graceful Fallback**: Falls back to MockProvider for testing when no API keys present
- **Tool Integration**: Connect to Mitochondria for native LLM function calling
- **Audit Trail**: Complete logging of all transcriptions with energy costs
- **Multi-turn Tool Loop**: Automatic tool execution with iteration limits

```python
from operon_ai import Nucleus, Mitochondria, ProviderConfig

# Auto-detects provider from environment variables
nucleus = Nucleus()
print(f"Using: {nucleus.provider.name}")  # anthropic, openai, gemini, or mock

# Simple transcription
response = nucleus.transcribe("Explain DNA replication")
print(response.content)

# Tool integration with Mitochondria
mito = Mitochondria(silent=True)
mito.register_function(
    name="calculator",
    func=lambda expr: str(mito.metabolize(expr).atp.value),
    description="Evaluate math expressions",
    parameters_schema={
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"]
    }
)

# LLM can now call tools automatically
response = nucleus.transcribe_with_tools(
    "What is 15 * 7 + 23? Use the calculator.",
    mitochondria=mito,
    config=ProviderConfig(temperature=0.0),
    max_iterations=5,
)
print(response.content)
```

### 🔌 LLM Providers

Swappable LLM backends with a unified interface:

| Provider | Model Default | Environment Variable | Features |
|----------|---------------|---------------------|----------|
| `AnthropicProvider` | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` | Tool use, streaming |
| `OpenAIProvider` | gpt-4o-mini | `OPENAI_API_KEY` | Tool use, JSON mode |
| `GeminiProvider` | gemini-flash-latest | `GEMINI_API_KEY` | Native function calling |
| `MockProvider` | mock | (none) | Testing, deterministic responses |

```python
from operon_ai import (
    Nucleus,
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    MockProvider,
)

# Explicit provider selection
nucleus = Nucleus(provider=GeminiProvider(model="gemini-flash-latest"))

# Or use auto-detection (checks env vars in order)
nucleus = Nucleus()  # Anthropic → OpenAI → Gemini → Mock
```

---

## 🧠 State Management

Biologically-inspired state systems for agents:

### 💊 Metabolism (Energy Management)

Multi-currency energy system with regeneration, debt, and sharing.

```python
from operon_ai.state import ATP_Store, MetabolicState, EnergyType

# Multiple energy currencies
metabolism = ATP_Store(
    budget=100,          # ATP for general operations
    gtp_budget=50,       # GTP for specialized tools
    nadh_reserve=30,     # NADH reserve (converts to ATP)
    regeneration_rate=5, # Regenerate 5 ATP/second
    max_debt=20,         # Allow energy debt
)

# Consume energy for operations
metabolism.consume(10, "llm_call", EnergyType.ATP)
metabolism.consume(20, "tool_use", EnergyType.GTP)

# Convert reserves when low
metabolism.convert_nadh_to_atp(15)

# Transfer energy between agents
metabolism.transfer_to(other_agent.metabolism, 10)
```

### 🧬 Genome (Immutable Configuration)

DNA-like configuration with gene expression control.

```python
from operon_ai.state import Genome, Gene, GeneType, ExpressionLevel

genome = Genome(genes=[
    Gene(name="model", value="gpt-4", gene_type=GeneType.STRUCTURAL, required=True),
    Gene(name="temperature", value=0.7, gene_type=GeneType.REGULATORY),
    Gene(name="debug_mode", value=True, gene_type=GeneType.CONDITIONAL),
])

# Silence a gene without changing it
genome.silence_gene("debug_mode")

# Get active configuration
config = genome.express()  # Only non-silenced genes

# Create child with mutations
child = genome.replicate(mutations={"temperature": 0.9})
```

### 🕰️ Telomere (Lifecycle Management)

Agent lifespan tracking with senescence and renewal.

```python
from operon_ai.state import Telomere, LifecyclePhase

telomere = Telomere(
    max_operations=1000,
    error_threshold=50,
    allow_renewal=True,
)

# Each operation shortens telomeres
while telomere.tick():  # Returns False when senescent
    do_operation()
    if telomere.record_error():
        handle_error()

# Renew agent (like telomerase)
if telomere.get_phase() == LifecyclePhase.SENESCENT:
    telomere.renew(amount=500)  # Extend lifespan
```

### 📝 Histone (Epigenetic Memory)

Multi-type memory with decay, reinforcement, and inheritance.

```python
from operon_ai.state import HistoneStore, MarkerType, MarkerStrength

memory = HistoneStore(enable_decay=True, decay_rate=0.1)

# Different marker types for different persistence
memory.methylate("user_preference", {"theme": "dark"},
                 marker_type=MarkerType.METHYLATION,  # Permanent
                 strength=MarkerStrength.STRONG)

memory.methylate("session_context", {"topic": "ML"},
                 marker_type=MarkerType.ACETYLATION,  # Temporary
                 tags=["context"])

# Semantic recall by tags
results = memory.retrieve_by_tags(["context"])

# Memory inheritance to child agents
child_memory = memory.create_child(inherit_methylations=True)
```

---

## 🔬 Network Topologies

Higher-order patterns that wire agents together:

### Coherent Feed-Forward Loop (CFFL)

**Two-key execution guardrail**: an action proceeds only if the executor and a risk assessor both permit it.

Note: the AND gate provides an interlock, not true independence—risk reduction depends on assessor diversity/tool-grounding.

```python
from operon_ai import ATP_Store
from operon_ai.topology import CoherentFeedForwardLoop, GateLogic

energy = ATP_Store(budget=100)
guardrail = CoherentFeedForwardLoop(
    budget=energy,
    gate_logic=GateLogic.AND,           # Both must agree
    enable_circuit_breaker=True,         # Prevent cascade failures
    enable_cache=True,                   # Cache decisions
)

result = guardrail.run("Deploy to production")
if result.blocked:
    print(f"Blocked: {result.block_reason}")
else:
    # Proof-carrying token bound to this request (two-key execution)
    print(f"Approval issuer: {result.approval_token.issuer}")
```

### Quorum Sensing

Multi-agent consensus with voting strategies and reliability tracking.

Note: quorum only helps if voters are not strongly correlated (use diverse models, tool checks, and/or data partitioning).

```python
from operon_ai.topology import QuorumSensing, VotingStrategy

quorum = QuorumSensing(
    n_agents=5,
    budget=budget,
    strategy=VotingStrategy.WEIGHTED,  # Weight-adjusted voting
)

# Set agent weights (experts have more influence)
quorum.set_agent_weight("Expert_Agent", 2.0)

result = quorum.run_vote("Should we deploy to production?")
print(f"Decision: {result.decision.value}")
print(f"Confidence: {result.confidence_score:.1%}")
```

### Cascade (Signal Amplification)

Multi-stage processing with checkpoints and amplification.

```python
from operon_ai.topology import Cascade, CascadeStage, MAPKCascade

cascade = Cascade(name="DataPipeline")

cascade.add_stage(CascadeStage(
    name="validate",
    processor=lambda x: x if valid(x) else None,
    checkpoint=lambda x: x is not None,  # Gate
))

cascade.add_stage(CascadeStage(
    name="transform",
    processor=lambda x: process(x),
    amplification=2.0,  # Signal amplification
))

result = cascade.run(input_data)
print(f"Amplification: {result.total_amplification}x")

# Or use pre-built MAPK cascade
mapk = MAPKCascade(tier1_amplification=10.0)
```

### Oscillator (Periodic Tasks)

Biological rhythms for scheduled operations.

```python
from operon_ai.topology import Oscillator, HeartbeatOscillator, OscillatorPhase

# Heartbeat for health checks
heartbeat = HeartbeatOscillator(
    beats_per_minute=60,
    on_beat=lambda: health_check(),
)
heartbeat.start()

# Custom work/rest cycle
osc = Oscillator(frequency_hz=0.1)  # 10 second period
osc.add_phase(OscillatorPhase(name="work", duration_seconds=7, action=do_work))
osc.add_phase(OscillatorPhase(name="rest", duration_seconds=3, action=do_rest))
osc.start()
```

---

## 🧷 Typed Wiring (WAgent)

Validate wiring diagrams up-front: type flow (`DataType`), integrity flow (`IntegrityLabel`), and aggregate effects (`Capability`).

```python
from operon_ai import (
    WiringDiagram, ModuleSpec, PortType,
    DataType, IntegrityLabel, Capability,
)

diagram = WiringDiagram()
diagram.add_module(ModuleSpec(
    name="membrane",
    outputs={"clean_text": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
))
diagram.add_module(ModuleSpec(
    name="executor",
    inputs={"prompt": PortType(DataType.TEXT, IntegrityLabel.VALIDATED)},
    capabilities={Capability.EXEC_CODE},
))

diagram.connect("membrane", "clean_text", "executor", "prompt")  # raises WiringError if invalid
print(diagram.required_capabilities())
```

## 📦 Installation

```bash
pip install operon-ai
```

## 🔬 Examples

Explore the `examples/` directory for runnable demonstrations:

### Basic Topologies

| Example | Pattern | Description |
|---------|---------|-------------|
| [`01_code_review_bot.py`](examples/01_code_review_bot.py) | CFFL | Dual-check guardrails (executor + risk assessor) |
| [`02_multi_model_consensus.py`](examples/02_multi_model_consensus.py) | Quorum | Multi-agent voting with threshold consensus |
| [`03_structured_extraction.py`](examples/03_structured_extraction.py) | Chaperone | Schema validation for raw text |
| [`04_budget_aware_agent.py`](examples/04_budget_aware_agent.py) | ATP | Resource management with graceful degradation |
| [`05_secure_chat_with_memory.py`](examples/05_secure_chat_with_memory.py) | Membrane+Histone | Input filtering + learned memory |
| [`06_sql_query_validation.py`](examples/06_sql_query_validation.py) | Chaperone | Domain-specific SQL validation |

### Advanced Organelles

| Example | Organelle | Description |
|---------|-----------|-------------|
| [`07_adaptive_membrane_defense.py`](examples/07_adaptive_membrane_defense.py) | Membrane | Adaptive immunity, antibody transfer, rate limiting |
| [`08_multi_pathway_mitochondria.py`](examples/08_multi_pathway_mitochondria.py) | Mitochondria | Safe AST computation, tool registry, ROS management |
| [`09_advanced_chaperone_folding.py`](examples/09_advanced_chaperone_folding.py) | Chaperone | Multi-strategy folding, confidence scoring |
| [`10_ribosome_prompt_factory.py`](examples/10_ribosome_prompt_factory.py) | Ribosome | Template synthesis, conditionals, loops, includes |
| [`11_lysosome_waste_management.py`](examples/11_lysosome_waste_management.py) | Lysosome | Cleanup, recycling, autophagy, toxic disposal |
| [`12_complete_cell_simulation.py`](examples/12_complete_cell_simulation.py) | **All** | Complete cellular lifecycle with all organelles |

### State Management & Topologies

| Example | System | Description |
|---------|--------|-------------|
| [`13_advanced_metabolism.py`](examples/13_advanced_metabolism.py) | Metabolism | Multi-currency (ATP/GTP/NADH), debt, regeneration, sharing |
| [`14_epigenetic_memory.py`](examples/14_epigenetic_memory.py) | Histone | Marker types, decay, reinforcement, inheritance |
| [`15_genome_telomere_lifecycle.py`](examples/15_genome_telomere_lifecycle.py) | Genome+Telomere | Immutable config, gene expression, lifecycle management |
| [`16_network_topologies.py`](examples/16_network_topologies.py) | Topologies | Cascade, Oscillator, enhanced QuorumSensing |
| [`17_wagent_typed_wiring.py`](examples/17_wagent_typed_wiring.py) | WAgent | Typed wiring checker (integrity + capabilities) |

### LLM Integration

| Example | System | Description |
|---------|--------|-------------|
| [`18_cell_integrity_demo.py`](examples/18_cell_integrity_demo.py) | Integrity | Quality, Surveillance, and Coordination systems |
| [`19_llm_code_assistant.py`](examples/19_llm_code_assistant.py) | Nucleus+CFFL | Code generation with two-phase safety review |
| [`20_llm_memory_chat.py`](examples/20_llm_memory_chat.py) | Nucleus+Histone | Conversational AI with epigenetic memory |
| [`21_llm_living_cell.py`](examples/21_llm_living_cell.py) | **Full Cell** | Complete lifecycle with LLM, memory, and aging |
| [`22_llm_tool_use.py`](examples/22_llm_tool_use.py) | Nucleus+Mitochondria | LLM function calling with tool integration |

Run any example:

```bash
# Clone and install
git clone https://github.com/coredipper/operon.git
cd operon
pip install -e .

# Run examples
python examples/07_adaptive_membrane_defense.py
python examples/12_complete_cell_simulation.py
```

---

## 📚 Theoretical Background

Operon models a functorial correspondence between Gene Regulatory Networks (GRNs) and agentic architectures, using typed wiring diagrams as the interface.

| Biological Concept | Software Equivalent | Mathematical Object |
|-------------------|---------------------|---------------------|
| Gene | Agent / System Prompt | Polynomial Functor (P) |
| Promoter | Context Schema | Lens (S→V) |
| Signal | Message / User Input | Type (T) |
| Epigenetics | RAG / Vector Store | State Monad (M) |
| Membrane | Input Filter | Predicate (T → Bool) |
| Ribosome | Template Engine | String Functor |
| Mitochondria | Tool Use | Effect Monad (capabilities) |
| Nucleus | LLM Provider | Decision Functor (prompt → response) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CELL (Agent)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  MEMBRANE   │───▶│   NUCLEUS   │───▶│MITOCHONDRIA │      │
│  │  (Filter)   │    │ (LLM/Think) │◀───│ (Tools)     │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                  │                  │             │
│         │                  ▼                  │             │
│         │           ┌─────────────┐           │             │
│         │           │  RIBOSOME   │           │             │
│         │           │ (Templates) │           │             │
│         │           └─────────────┘           │             │
│         │                  │                  │             │
│         │                  ▼                  ▼             │
│         │           ┌─────────────┐    ┌─────────────┐      │
│         │           │  CHAPERONE  │    │   HISTONE   │      │
│         │           │ (Validate)  │    │  (Memory)   │      │
│         │           └─────────────┘    └─────────────┘      │
│         │                  │                                │
│         ▼                  ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                      LYSOSOME                       │    │
│  │              (Cleanup & Recycling)                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

We are looking for contributors to build out the Plasmid Registry (a marketplace of dynamic tools) and expand the Quorum Sensing algorithms.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/plasmid-loading`)
3. Commit your changes
4. Open a Pull Request

## 📄 License

Distributed under the MIT License. See LICENSE for more information.
