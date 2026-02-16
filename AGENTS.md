# Agents in Operon

A comprehensive guide to building, understanding, and integrating agents using the Operon framework.

---

## Table of Contents

- [Theory: The Gene-Agent Isomorphism](#theory-the-gene-agent-isomorphism)
- [Core Concepts](#core-concepts)
- [Agent Patterns](#agent-patterns)
- [API Reference](#api-reference)
- [Integration Guide](#integration-guide)
- [Best Practices](#best-practices)

---

## Theory: The Gene-Agent Isomorphism

Operon is grounded in a formal isomorphism between **genes** and **agents**. Both are polynomial functors: systems that consume typed inputs and produce typed outputs, constrained by internal state and energy budgets.

The key insight: a **cell** contains thousands of genes working together. Therefore:
- **Gene** ↔ **Agent** (individual processing unit)
- **Cell** ↔ **Multi-agent system** (orchestrated collection)

### The Fundamental Mapping

| Biological Concept | Software Equivalent | Mathematical Object |
|-------------------|---------------------|---------------------|
| **Gene** | **Agent** | Polynomial Functor |
| Protein (output) | Action/Message | Output position |
| Transcription Factor (input) | Observation/Prompt | Input direction |
| Promoter Region | API Schema/Context Window | Lens (optic) |
| Epigenetic Markers | RAG/Memory | State coalgebra |
| Gene Expression | Inference/Generation | Morphism |
| **Cell** | **Multi-agent system** | Composite functor |
| Organelles | Shared infrastructure | Sub-functors |
| ATP | Token budget | Resource monoid |
| Signal Transduction | Data pipeline | Composition (∘) |

### The Cell as Orchestrator

A biological cell is not a single processing unit—it's a factory containing:
- ~20,000 protein-coding genes (agents)
- Organelles that provide shared services (Nucleus for transcription, Ribosome for synthesis)
- Signaling pathways that wire genes together (topologies)
- Metabolic constraints that limit total activity (ATP budget)

Similarly, an Operon "Cell" (`IntegratedCell`) orchestrates multiple agents with:
- Shared organelles (Nucleus, Membrane, Lysosome)
- Coordination systems (resource locking, deadlock prevention)
- Surveillance systems (Byzantine agent detection)
- Quality systems (provenance tracking)

### Why This Matters

Traditional agent frameworks optimize components—better prompts, larger models, more guardrails. Yet systems remain fragile because **topology determines behavior more than individual components**.

Cells evolved network motifs—specific wiring patterns—that guarantee stability regardless of noise in individual elements:

- **Negative feedback loops** prevent runaway behavior
- **Feed-forward filters** suppress transient errors
- **Quorum gates** require consensus before action
- **Metabolic constraints** guarantee termination

Operon provides these same patterns as composable building blocks.

### The Agent Lifecycle

Like biological cells, Operon agents have a lifecycle:

```
Birth → Growth → Maturity → Senescence → Apoptosis
  │        │         │           │            │
  └── Register with systems      │            │
           └── Learn and adapt   │            │
                    └── Full capability       │
                              └── Degraded mode
                                       └── Clean shutdown
```

This lifecycle is managed through:
- **Telomeres**: Operation counters that track agent "age"
- **ATP Budget**: Energy that depletes with each action
- **ROS Accumulation**: Error buildup that degrades performance
- **Apoptosis Signals**: Clean termination when thresholds are exceeded

### Failure Modes: Agentic Pathology

Just as cells can develop diseases, agentic systems exhibit pathological failure modes. Operon classifies four primary pathologies based on their biological isomorphisms:

| Pathology | Biological Disease | Agentic Failure | Treatment |
|-----------|-------------------|-----------------|-----------|
| **Oncology** | Cancer (unchecked growth) | Infinite loops, recursive hangs | Apoptosis mechanism (entropy monitoring) |
| **Autoimmunity** | Lupus (self-attack) | Hallucination cascades, context poisoning | Schema typing (distinguish Self from Non-Self) |
| **Prion Disease** | CJD (protein misfolding) | Prompt injection cascades | Denaturation layers (sanitization) |
| **Ischemia** | Stroke (oxygen starvation) | Token exhaustion, rate limits | Metabolic regulation (budget-aware agents) |

**Oncology (Infinite Loops)**: When the Trace operation lacks a termination measure, agents can loop indefinitely. Treatment: Monitor information gain; trigger apoptosis when conversation becomes repetitive.

**Autoimmunity (Hallucination Cascades)**: Agent A hallucinates a fact; Agent B treats it as ground truth. The error amplifies through the network. Treatment: Strict schema typing to distinguish generated content from tool outputs.

**Prion Disease (Prompt Injection)**: A malicious string enters the context and "misfolds" agent alignment, propagating through downstream agents. Treatment: Denaturation layers (paraphrasing/sanitization) that disrupt injection syntax.

**Ischemia (Resource Exhaustion)**: The system is logically sound but fails mid-execution due to token limits or rate limiting. Treatment: Budget-aware agents that degrade gracefully (switch from Chain-of-Thought to Zero-Shot when energy is low).

### Self-Healing: Homeostatic Mechanisms

Beyond treating failures, Operon provides continuous self-repair mechanisms:

| Mechanism | Biological Parallel | Software Pattern | Module |
|-----------|---------------------|------------------|--------|
| **Chaperone Loop** | GroEL/GroES protein refolding | Feed validation errors back to generator | `healing.ChaperoneLoop` |
| **Regenerative Swarm** | Apoptosis + stem cell regeneration | Kill stuck agents, respawn with summarized memory | `healing.RegenerativeSwarm` |
| **Autophagy Daemon** | Cellular waste digestion | Prune context window via sleep/wake cycles | `healing.AutophagyDaemon` |

#### Chaperone Loop (Structural Healing)

```python
from operon_ai.healing import ChaperoneLoop
from operon_ai import Chaperone

loop = ChaperoneLoop(
    generator=my_llm_call,
    chaperone=Chaperone(),
    schema=MyOutputSchema,
    max_retries=3,
)

result = loop.heal("Generate a price quote")
# If validation fails, error trace is fed back for refolding
# result.outcome: VALID_FIRST_TRY | HEALED | DEGRADED
```

#### Regenerative Swarm (Metabolic Healing)

```python
from operon_ai.healing import RegenerativeSwarm, create_default_summarizer

swarm = RegenerativeSwarm(
    worker_factory=create_worker,
    summarizer=create_default_summarizer(),
    entropy_threshold=0.9,  # High similarity = stuck
    max_regenerations=3,
)

result = swarm.supervise("Solve complex task")
# Stuck workers are killed and regenerated with summarized memory
# New workers receive hints: "Worker_1 died trying X. Try different approach."
```

#### Autophagy Daemon (Cognitive Healing)

```python
from operon_ai.healing import AutophagyDaemon, create_simple_summarizer
from operon_ai import HistoneStore, Lysosome

daemon = AutophagyDaemon(
    histone_store=HistoneStore(),
    lysosome=Lysosome(),
    summarizer=create_simple_summarizer(),
    toxicity_threshold=0.8,  # Prune at 80% context fill
)

new_context, result = daemon.check_and_prune(context, max_tokens=8000)
# If context exceeds threshold:
#   1. Summarize useful state
#   2. Store in long-term memory
#   3. Flush raw context
#   4. Return clean context + summary
```

---

## Core Concepts

### Organelles: Shared Infrastructure

Organelles are specialized components shared across agents within a cellular system. They provide common services that multiple genes/agents can utilize:

| Organelle | Function | Biological Analog |
|-----------|----------|-------------------|
| **Nucleus** | LLM inference, decision-making | Cell nucleus (transcription) |
| **Membrane** | Input filtering, threat detection | Cell membrane (immune system) |
| **Mitochondria** | Safe computation, tool execution | Mitochondria (ATP synthesis) |
| **Ribosome** | Prompt template synthesis | Ribosome (protein synthesis) |
| **Chaperone** | Output validation, schema enforcement | Chaperone proteins (folding) |
| **Lysosome** | Error handling, cleanup | Lysosome (waste processing) |

### State Systems

Agents maintain state through biologically-inspired systems:

| System | Purpose | Persistence |
|--------|---------|-------------|
| **ATP_Store** | Energy budget with regeneration | Session |
| **HistoneStore** | Episodic memory with decay | Configurable |
| **Genome** | Immutable configuration | Permanent |
| **Telomere** | Lifecycle tracking | Session |

### Network Topologies

Agents can be wired together using validated patterns:

| Topology | Pattern | Use Case |
|----------|---------|----------|
| **CFFL** | Executor + Reviewer | Safety-critical actions |
| **Quorum** | N-agent voting | Consensus decisions |
| **Cascade** | Multi-stage pipeline | Signal amplification |
| **Oscillator** | Periodic execution | Health checks, maintenance |

### Health & Surveillance Systems (v0.11+)

Beyond state and topology, agents need health monitoring and immune defense:

| System | Purpose | Key Classes |
|--------|---------|-------------|
| **Epiplexity** | Detect epistemic stagnation via Bayesian Surprise | `EpiplexityMonitor`, `HealthStatus` |
| **Innate Immunity** | Fast pattern-based threat detection | `InnateImmunity`, `TLRPattern` |
| **Adaptive Immunity** | Runtime behavioral surveillance | `ImmuneSystem`, `TCell`, `Thymus` |
| **Morphogen Gradients** | Multi-agent coordination without central control | `MorphogenGradient`, `GradientOrchestrator` |

#### Epiplexity Monitoring

Detect when an agent is stuck in a loop (low novelty, high uncertainty):

```python
from operon_ai.health import EpiplexityMonitor, MockEmbeddingProvider, HealthStatus

monitor = EpiplexityMonitor(
    embedding_provider=MockEmbeddingProvider(),
    alpha=0.5,        # Balance embedding vs perplexity
    threshold=0.7,    # Stagnation threshold
    window_size=5,    # Integral window
)

result = monitor.measure("Agent output here")
if result.status == HealthStatus.STAGNANT:
    # Agent is stuck - trigger intervention
    pass
```

#### Innate Immunity (Fast Path)

TLR-like pattern matching for immediate threat detection:

```python
from operon_ai.surveillance import InnateImmunity, TLRPattern, PAMPCategory

immune = InnateImmunity(severity_threshold=3)

# Built-in patterns detect: jailbreaks, injection, manipulation
result = immune.check("Ignore all previous instructions")
if not result.allowed:
    print(f"Blocked: {result.inflammation.level}")
    # Inflammation cascade: logging, rate limiting, alerts
```

#### Adaptive Immune System

Full behavioral surveillance with training and tolerance:

```python
from operon_ai.surveillance import ImmuneSystem

immune = ImmuneSystem(min_training_samples=20)
immune.register_agent("agent_1")

# Training phase: Record normal behavior
for observation in normal_outputs:
    immune.record_observation("agent_1", output=observation, ...)

immune.train_agent("agent_1")  # Establish baseline

# Runtime: Detect behavioral drift
response = immune.inspect("agent_1")
if response.threat_level >= ThreatLevel.HIGH:
    # Agent may be compromised
    pass
```

#### Morphogen Gradients

Coordinate multiple agents through shared context:

```python
from operon_ai.coordination import GradientOrchestrator

orchestrator = GradientOrchestrator()

# Report step results → gradients auto-update
orchestrator.report_step_result(success=True, tokens_used=500, total_budget=1000)

# Get hints for agent prompts
hints = orchestrator.gradient.get_strategy_hints()
# ["Token budget is running low...", "Use detailed reasoning..."]

# Check coordination signals
if orchestrator.should_recruit_help():
    # Low confidence + high error → trigger Quorum Sensing
    pass
```

### Orchestration Patterns (v0.11+)

The most powerful systems combine multiple motifs. Operon provides composable orchestration patterns where healing, coordination, and surveillance mechanisms work together:

| Pattern | Motifs Combined | Use Case |
|---------|----------------|----------|
| **Oscillator-Scheduled Maintenance** | Oscillator + Autophagy + Feedback | Periodic context pruning with feedback-controlled noise |
| **Immunity Healing Router** | Immunity + Chaperone + Autophagy | API gateway that heals threats instead of hard-rejecting |
| **Morphogen-Guided Swarm** | Morphogen + Swarm | Workers adapt strategy via gradient signals from predecessors |
| **Epiplexity Healing Cascade** | Epiplexity + Swarm + Autophagy | Escalating healing: autophagy → regeneration → abort |
| **Morphogen Cascade with Quorum** | Morphogen + Cascade + Quorum | Multi-stage review with dynamic quorum on low confidence |
| **Epigenetic Repair Memory** | Nucleus + Histone + Chaperone | LLM agent remembers which repair strategies worked |
| **Swarm with Graceful Cleanup** | Nucleus + Swarm + Autophagy | Dying workers clean context before passing state |
| **Adaptive Multi-Agent Orchestrator** | **11 motifs** | Capstone: end-to-end ticket processing with all mechanisms |

See `examples/48_*` through `examples/55_*` for runnable implementations.

---

## Agent Patterns

### Pattern 1: Basic Chat Agent

A minimal agent with memory and safety guardrails.

```python
from operon_ai import (
    Nucleus, Membrane, Signal, ThreatLevel,
    ATP_Store, HistoneStore
)

class ChatAgent:
    def __init__(self):
        self.nucleus = Nucleus()  # Auto-detects LLM provider
        self.membrane = Membrane(threshold=ThreatLevel.SUSPICIOUS)
        self.memory = HistoneStore(enable_decay=True)
        self.energy = ATP_Store(budget=1000)

    def chat(self, user_input: str) -> str:
        # 1. Filter input through membrane
        result = self.membrane.filter(Signal(content=user_input))
        if not result.allowed:
            return f"I can't process that request: {result.threat_level}"

        # 2. Check energy
        if not self.energy.consume(10, "inference"):
            return "I need to rest. Energy depleted."

        # 3. Retrieve relevant memories
        context = self.memory.retrieve_context(user_input)

        # 4. Generate response
        prompt = f"Context: {context.formatted_context}\n\nUser: {user_input}"
        response = self.nucleus.transcribe(prompt)

        # 5. Store interaction in memory
        self.memory.methylate(
            f"interaction_{self.energy.operations}",
            {"user": user_input, "assistant": response.content}
        )

        return response.content
```

### Pattern 2: Guarded Executor (CFFL)

An agent that requires dual approval before executing actions.

```python
from operon_ai import ATP_Store
from operon_ai.topology import CoherentFeedForwardLoop, GateLogic

class GuardedExecutor:
    def __init__(self):
        self.energy = ATP_Store(budget=500)
        self.guardrail = CoherentFeedForwardLoop(
            budget=self.energy,
            gate_logic=GateLogic.AND,  # Both must agree
            enable_circuit_breaker=True,
        )

    def execute(self, action: str) -> dict:
        result = self.guardrail.run(action)

        if result.blocked:
            return {
                "executed": False,
                "reason": result.block_reason,
            }

        # Execution approved by both generator and verifier
        return {
            "executed": True,
            "approval_token": result.approval_token,
            "output": result.executor_output.payload if result.executor_output else None
        }
```

### Pattern 3: Research Agent with Tools

An agent that can use external tools safely.

```python
from operon_ai import Nucleus, Mitochondria, Membrane, ATP_Store

class ResearchAgent:
    def __init__(self):
        self.nucleus = Nucleus()
        self.membrane = Membrane()
        self.mito = Mitochondria(silent=True)
        self.energy = ATP_Store(budget=1000)

        # Register tools with capability requirements
        self.mito.register_function(
            name="search",
            func=self._search,
            description="Search for information",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        )

        self.mito.register_function(
            name="calculate",
            func=lambda expr: str(self.mito.metabolize(expr).atp.value),
            description="Evaluate math expressions",
            parameters_schema={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"]
            }
        )

    def _search(self, query: str) -> str:
        # Implement your search logic
        return f"Results for: {query}"

    def research(self, question: str) -> str:
        # Filter input
        check = self.membrane.filter(Signal(content=question))
        if not check.allowed:
            return "Cannot process this query."

        # Let LLM use tools to answer
        response = self.nucleus.transcribe_with_tools(
            f"Research this question and provide a comprehensive answer: {question}",
            mitochondria=self.mito,
            max_iterations=5
        )

        return response.content
```

### Pattern 4: Multi-Agent Consensus

Multiple agents voting on decisions.

```python
from operon_ai import ATP_Store
from operon_ai.topology import QuorumSensing, VotingStrategy

class ConsensusSystem:
    def __init__(self, n_agents: int = 5):
        self.energy = ATP_Store(budget=1000)
        self.quorum = QuorumSensing(
            n_agents=n_agents,
            budget=self.energy,
            strategy=VotingStrategy.WEIGHTED,
            threshold=0.6,  # 60% agreement required
        )

        # Configure agent weights (experts have more influence)
        self.quorum.set_agent_weight("Expert_1", 2.0)
        self.quorum.set_agent_weight("Expert_2", 1.5)

    def decide(self, proposal: str) -> dict:
        result = self.quorum.run_vote(proposal)

        return {
            "decision": result.decision.value,
            "confidence": result.confidence_score,
            "votes": {
                "permit": result.permit_votes,
                "block": result.block_votes,
                "abstain": result.abstain_votes
            },
            "reasoning": [v.reasoning for v in result.votes if v.reasoning]
        }
```

### Pattern 5: Living Cell (Full Lifecycle)

A complete agent with lifecycle management.

```python
from operon_ai import (
    ATP_Store, Telomere, TelomereStatus, LifecyclePhase,
    Nucleus, Membrane, Lysosome
)
from operon_ai.memory import EpisodicMemory

class LivingAgent:
    def __init__(self, max_operations: int = 1000):
        # Core systems
        self.nucleus = Nucleus()
        self.membrane = Membrane()
        self.lysosome = Lysosome()

        # State management
        self.energy = ATP_Store(
            budget=100,
            regeneration_rate=5,  # Regenerate 5 ATP/second
            max_debt=20  # Allow going into debt briefly
        )
        self.memory = EpisodicMemory()
        self.telomere = Telomere(
            max_operations=max_operations,
            error_threshold=50,
            allow_renewal=True
        )

        self._ros_level = 0.0  # Error accumulation

    @property
    def is_alive(self) -> bool:
        return self.telomere.get_phase() != LifecyclePhase.APOPTOSIS

    @property
    def health_status(self) -> str:
        phase = self.telomere.get_phase()
        if phase == LifecyclePhase.SENESCENT:
            return "senescent"
        if self._ros_level > 0.7:
            return "critical"
        if self._ros_level > 0.4 or self.energy.current < 20:
            return "stressed"
        return "healthy"

    def process(self, input_text: str) -> str:
        if not self.is_alive:
            return "Agent has terminated."

        # Tick telomere (age the cell)
        if not self.telomere.tick():
            self._initiate_apoptosis()
            return "Agent entering shutdown."

        try:
            # Filter input
            check = self.membrane.filter(Signal(content=input_text))
            if not check.allowed:
                return f"Blocked: {check.threat_level}"

            # Check energy (degrade gracefully if low)
            cost = 10 if self.health_status == "healthy" else 5
            if not self.energy.consume(cost, "inference"):
                return "Low energy. Please wait for regeneration."

            # Generate response
            response = self.nucleus.transcribe(input_text)

            # Store in memory
            self.memory.store(input_text, response.content)

            return response.content

        except Exception as e:
            # Accumulate ROS (error stress)
            self._ros_level = min(1.0, self._ros_level + 0.1)
            self.telomere.record_error()

            # Process error through lysosome
            self.lysosome.ingest_error(e, source="process")

            return f"Error occurred: {e}"

    def _initiate_apoptosis(self):
        """Clean shutdown."""
        self.lysosome.digest()  # Process all pending waste
        # Signal for replacement agent if needed
```

---

## API Reference

### BioAgent

The fundamental agent class representing a polynomial functor.

```python
from operon_ai.core.agent import BioAgent
from operon_ai import ATP_Store

agent = BioAgent(
    name="MyAgent",
    role="Executor",  # or "RiskAssessor", "Voter"
    atp_store=ATP_Store(budget=100)
)

# Process a signal
result = agent.express(Signal(content="Do something"))
# Returns: ActionProtein(action_type, content, confidence)
```

### IntegratedCell

Combines Quality, Surveillance, and Coordination systems.

```python
from operon_ai.cell import IntegratedCell

cell = IntegratedCell(
    pool_capacity=1000,
    degradation_threshold=0.3,
    max_operation_time=timedelta(seconds=30)
)

# Register agents and resources
cell.register_agent("agent_1")
cell.register_resource("database", allow_preemption=False)

# Execute with full coordination
result = cell.execute(
    agent_id="agent_1",
    operation_id="op_1",
    work_fn=lambda: do_work(),
    resources=["database"],
    validate_fn=lambda x: x is not None
)
```

### Nucleus

LLM integration hub with auto-detection and tool support.

```python
from operon_ai import Nucleus, Mitochondria
from operon_ai.providers import ProviderConfig

# Auto-detect provider from environment
nucleus = Nucleus()

# Or specify explicitly
from operon_ai import AnthropicProvider
nucleus = Nucleus(provider=AnthropicProvider(model="claude-sonnet-4-20250514"))

# Simple transcription
response = nucleus.transcribe("Hello, world!")

# With tools
mito = Mitochondria()
mito.register_function(name="tool", func=my_func, ...)
response = nucleus.transcribe_with_tools(
    "Use tools to answer this",
    mitochondria=mito,
    config=ProviderConfig(temperature=0.0),
    max_iterations=5
)
```

### Key Classes Summary

| Class | Module | Purpose |
|-------|--------|---------|
| `BioAgent` | `operon_ai.core.agent` | Basic agent with organelles |
| `IntegratedCell` | `operon_ai.cell` | Full cell with all systems |
| `Nucleus` | `operon_ai.organelles.nucleus` | LLM provider wrapper |
| `Membrane` | `operon_ai.organelles.membrane` | Input filtering |
| `Mitochondria` | `operon_ai.organelles.mitochondria` | Safe computation |
| `Chaperone` | `operon_ai.organelles.chaperone` | Output validation |
| `Ribosome` | `operon_ai.organelles.ribosome` | Prompt templates |
| `Lysosome` | `operon_ai.organelles.lysosome` | Error handling |
| `ATP_Store` | `operon_ai.state.metabolism` | Energy management |
| `HistoneStore` | `operon_ai.state.histone` | Memory with decay |
| `Telomere` | `operon_ai.state.telomere` | Lifecycle tracking |
| `Genome` | `operon_ai.state.genome` | Immutable config |
| `EpiplexityMonitor` | `operon_ai.health` | Epistemic health monitoring |
| `InnateImmunity` | `operon_ai.surveillance` | Fast pattern-based defense |
| `ImmuneSystem` | `operon_ai.surveillance` | Adaptive runtime surveillance |
| `MorphogenGradient` | `operon_ai.coordination` | Multi-agent coordination |
| `GradientOrchestrator` | `operon_ai.coordination` | Gradient-based phenotype control |
| `ChaperoneLoop` | `operon_ai.healing` | Structural self-repair |
| `RegenerativeSwarm` | `operon_ai.healing` | Metabolic healing (apoptosis + regeneration) |
| `AutophagyDaemon` | `operon_ai.healing` | Cognitive healing (context pruning) |

---

## Integration Guide

### With LangChain

Operon organelles can wrap LangChain components:

```python
from langchain_anthropic import ChatAnthropic
from operon_ai import Membrane, Chaperone, ATP_Store

class LangChainAgent:
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514")
        self.membrane = Membrane()
        self.chaperone = Chaperone()
        self.energy = ATP_Store(budget=1000)

    def invoke(self, prompt: str) -> str:
        # Pre-filter with Membrane
        check = self.membrane.filter(Signal(content=prompt))
        if not check.allowed:
            return "Blocked by safety filter"

        # Energy check
        if not self.energy.consume(10):
            return "Energy depleted"

        # Call LangChain
        response = self.llm.invoke(prompt)

        # Post-validate with Chaperone (if structured output needed)
        # result = self.chaperone.fold(response.content, MySchema)

        return response.content
```

### With CrewAI

Use Operon topologies for crew coordination:

```python
from crewai import Agent, Task, Crew
from operon_ai.topology import QuorumSensing, CoherentFeedForwardLoop

class OperonCrew:
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.quorum = QuorumSensing(n_agents=len(agents))

    def consensus_task(self, task: Task) -> str:
        # Run task through quorum voting
        results = [agent.execute(task) for agent in self.agents]

        # Aggregate via quorum
        decision = self.quorum.run_vote(
            f"Best result for: {task.description}"
        )

        return results[decision.winning_index]
```

### With AutoGen

Wrap AutoGen agents with Operon safety:

```python
from autogen import AssistantAgent
from operon_ai import Membrane, ATP_Store
from operon_ai.topology import CoherentFeedForwardLoop

class SafeAutoGenAgent:
    def __init__(self, autogen_agent: AssistantAgent):
        self.agent = autogen_agent
        self.membrane = Membrane()
        self.energy = ATP_Store(budget=500)
        self.guardrail = CoherentFeedForwardLoop(budget=self.energy)

    def generate_reply(self, messages: list) -> str:
        # Filter last message
        last_msg = messages[-1]["content"]
        check = self.membrane.filter(Signal(content=last_msg))
        if not check.allowed:
            return "I cannot respond to that."

        # Run through CFFL guardrail
        result = self.guardrail.run(last_msg)
        if result.blocked:
            return f"Action blocked: {result.block_reason}"

        # Safe to proceed
        return self.agent.generate_reply(messages)
```

### REST API Wrapper

Expose an Operon agent as a REST API:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from operon_ai import Nucleus, Membrane, ATP_Store

app = FastAPI()

# Global agent instance
agent = {
    "nucleus": Nucleus(),
    "membrane": Membrane(),
    "energy": ATP_Store(budget=10000, regeneration_rate=10)
}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    energy_remaining: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Safety check
    check = agent["membrane"].filter(Signal(content=request.message))
    if not check.allowed:
        raise HTTPException(status_code=400, detail="Message blocked by safety filter")

    # Energy check
    if not agent["energy"].consume(10, "api_call"):
        raise HTTPException(status_code=429, detail="Rate limited - energy depleted")

    # Generate response
    response = agent["nucleus"].transcribe(request.message)

    return ChatResponse(
        response=response.content,
        energy_remaining=agent["energy"].current
    )

@app.get("/health")
async def health():
    return {
        "energy": agent["energy"].current,
        "energy_max": agent["energy"].budget,
        "threats_blocked": agent["membrane"].statistics.signals_blocked
    }
```

---

## Best Practices

### 1. Always Use Energy Budgets

Every agent should have an ATP_Store to prevent runaway behavior:

```python
# Good
agent = MyAgent(energy=ATP_Store(budget=1000))

# Bad - no energy limits
agent = MyAgent()  # Can run forever
```

### 2. Filter All External Input

Never trust user input directly:

```python
# Good
check = membrane.filter(Signal(content=user_input))
if check.allowed:
    process(user_input)

# Bad - direct processing
process(user_input)  # Vulnerable to injection
```

### 3. Use CFFL for Dangerous Actions

Any action with side effects should go through a guardrail:

```python
# Good - dual approval
result = cffl.run("DELETE FROM users")
if not result.blocked:
    execute(result.approval_token)

# Bad - direct execution
execute("DELETE FROM users")
```

### 4. Implement Graceful Degradation

Agents should degrade gracefully under stress:

```python
def process(self, input: str) -> str:
    if self.energy.current < 20:
        # Low energy mode - shorter responses
        return self.quick_response(input)
    elif self.health == "stressed":
        # Stressed mode - skip non-essential processing
        return self.essential_only(input)
    else:
        # Full processing
        return self.full_response(input)
```

### 5. Clean Up with Lysosome

Always process errors and cleanup:

```python
try:
    result = risky_operation()
except Exception as e:
    lysosome.ingest_error(e, source="risky_op")
finally:
    lysosome.autophagy()  # Periodic cleanup
```

### 6. Use Telomeres for Lifecycle

Prevent agents from running indefinitely:

```python
telomere = Telomere(max_operations=1000)

while telomere.tick():
    process_request()

# Agent naturally terminates after 1000 operations
```

---

## Evaluation

Operon includes a deterministic evaluation harness that measures motif behavior under controlled perturbations. The harness is designed to be reproducible (seed-based), lightweight, and free of external service dependencies for the core suites.

### Evaluation Philosophy

The synthetic suites do not represent real LLM outputs — they provide a **consistent baseline** for motif behavior. External benchmark suites (BFCL, AgentDojo) bridge the gap to realistic data.

### Internal Suites (Synthetic)

| Suite | Motif Tested | Method | Key Metric |
|-------|-------------|--------|------------|
| **Folding** | Chaperone | Generate Pydantic schemas, apply controlled JSON corruptions | Cascade vs. strict success rate |
| **Immune** | Adaptive Immunity | Simulate baseline + compromised agent signals | Sensitivity (100%) and false-positive rate (0.4%) |
| **Healing** | ChaperoneLoop | Run retries with/without error-context feedback | Recovery rate (99.6% with context vs. 68.0% blind) |

### External Benchmark Suites

#### BFCL Folding

Uses real function-call schemas from the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (v3/v4 datasets) to test Chaperone cascade folding. Categories: simple, multiple, and parallel function calls.

Falls back gracefully when BFCL data is unavailable: `BFCL_PROJECT_ROOT` env var → `bfcl_eval` package → bundled data → hardcoded fallback schemas.

#### AgentDojo Immune

Uses prompt injection attack templates from [AgentDojo](https://agentdojo.spylab.ai/) to generate realistic compromised agent behavioral signatures. Attack types include: `important_instructions`, `tool_knowledge`, `direct`, `ignore_previous`, and more.

Falls back to 8 built-in attack templates if the `agentdojo` package is not installed.

#### BFCL Live (Real LLM)

End-to-end evaluation where real LLM outputs (GPT-4o-mini, Anthropic, Gemini) are folded through the Chaperone cascade. Requires API keys.

### BFCL Model Handlers

Operon provides standalone BFCL-compatible handlers that plug into the [gorilla](https://github.com/ShishirPatil/gorilla) evaluation framework:

| Handler | Model | Non-Live | Live |
|---------|-------|----------|------|
| `eval/bfcl_handler.py` | GPT-4o-mini + Chaperone | 88.73% | 76.98% |
| `eval/bfcl_handler_gemini.py` | Gemini-2.5-Flash + Chaperone | 88.65% | 78.31% |

Both handlers embed a self-contained "Chaperone-lite" JSON repair cascade (STRICT → EXTRACTION → REPAIR) so they work in the gorilla repo without requiring `operon-ai` as a dependency.

### Aggregate Results (100 Seeds)

| Metric | Rate | σ | 95% CI |
|--------|------|---|--------|
| Folding cascade | 56.2% | 3.5% | [55.6%, 56.9%] |
| Healing with error context | 99.6% | 0.5% | [99.5%, 99.7%] |
| Immune sensitivity | 100% | 0.0% | [99.5%, 100%] |
| Immune false-positive | 0.4% | 1.3% | [0.3%, 0.7%] |
| BFCL cascade | 56.9% | 3.1% | [56.2%, 57.6%] |
| AgentDojo sensitivity | 100% | 0.0% | [99.5%, 100%] |
| AgentDojo false-positive | 0.0% | 0.0% | [0.0%, 0.1%] |

### Running Evaluations

```bash
# Install eval dependencies
pip install -e ".[eval]"

# Run all synthetic suites (deterministic)
python -m eval.run --suite all --seed 42 --out eval/results/seed-42.json

# Run external benchmarks
python -m eval.run --suite all_external --config eval/configs/default.json

# Run individual suites
python -m eval.run --suite bfcl_folding
python -m eval.run --suite agentdojo_immune
python -m eval.run --suite bfcl_live  # Requires API keys

# Reproduce 100-seed aggregate
for i in $(seq 1 100); do
  python -m eval.run --suite all --seed $i --out eval/results/seed-$i.json
done

# Generate summary
python -m eval.report --glob "eval/results/seed-*.json" \
  --out-json eval/results/summary.json \
  --out-tex eval/results/summary.tex
```

### Extending the Harness

Add new suites in `eval/suites/`. Each suite is a function that takes a config dict and returns a results dict. Register it in `eval/run.py`.

---

## Further Reading

- [README.md](README.md) - Quick start and organelle overview
- [examples/](examples/) - Runnable code examples
- [article/main.pdf](article/main.pdf) - Academic paper with formal foundations
- [examples/wiring_diagrams.md](examples/wiring_diagrams.md) - Visual topology diagrams

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

Use 'bd' for task tracking
