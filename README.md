# Operon 🧬

**Biologically Inspired Architectures for Agentic Control**

> *"Don't fix the prompt. Fix the topology."*

![Status](https://img.shields.io/badge/status-experimental-orange)
![Version](https://img.shields.io/badge/pypi-v0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> ⚠️ **Note:** Operon is a research-grade library serving as the reference implementation for the paper *"Biological Motifs for Agentic Control."* APIs are subject to change as the theoretical framework evolves.

---

## 🦠 The Problem: Fragile Agents

Most agentic systems today are built like **cancerous cells**: they lack negative feedback loops, suffer from unchecked recursion (infinite loops), and are easily hijacked by foreign signals (prompt injection). We try to fix this with "Prompt Engineering"—optimizing the internal state.

**Biology solved this billions of years ago.** Cells don't rely on a central CPU; they rely on **Network Motifs**—specific wiring diagrams that guarantee robustness, consensus, and safety regardless of the noise in individual components.

**Operon** brings these biological control structures to Python. It uses **Applied Category Theory** to define rigorous "wiring diagrams" for agents, ensuring your system behaves like a multicellular organism, not a soup of stochastic scripts.

---

## 🧩 Features: The Biological Stack

We map biological survival mechanisms directly to software design patterns.

* **🛡️ Prion Defense (Membrane):**
    * **Biology:** Prevents misfolded proteins (prions) from infecting the cell.
    * **Code:** Input sanitization layers that detect and block **Prompt Injections** before they ever reach the Agent's context window.

* **⚡ Mitochondrial Tools (Endosymbiosis):**
    * **Biology:** Organelles that provide massive energy (ATP) to the host.
    * **Code:** **Neuro-symbolic** execution environments. Your LLM (Host) "engulfs" a Python REPL (Mitochondria) to perform exact math and logic, trading raw tokens for deterministic truth.

* **🧬 Epigenetic State (Histones):**
    * **Biology:** Chemical markers that silence genes based on past stress, without changing DNA.
    * **Code:** **Stateful Memory** that biases agent behavior. If an agent crashes on a specific task, it "methylates" that context, preventing it from retrying the same doomed action in the future.

* **🧶 Chaperone Validation (Folding):**
    * **Biology:** Proteins that force linear amino acid chains to fold into functional 3D shapes.
    * **Code:** **Retry loops** that force raw LLM text into strict **Pydantic/JSON schemas**. If the output "misfolds" (invalid JSON), the Chaperone rejects it and triggers a retry.

---

## 📦 Installation

```bash
pip install operon-ai
```

## 🧪 Usage
1. The Coherent Feed-Forward Loop (CFFL)
The "Human-in-the-Loop" Guardrail.

In biology, this circuit filters out noise. In software, it ensures an Executor agent cannot act unless a RiskAssessor independently agrees.

```python
from operon import ATP_Store, CoherentFeedForwardLoop

# Initialize metabolic budget (100 tokens)
energy = ATP_Store(budget=100)

# Create the topology
# Logic: Executor (Z) AND Assessor (Y) -> Action
guardrail = CoherentFeedForwardLoop(budget=energy)

# Scenario: Dangerous Request
# The Executor wants to run it, but the Assessor blocks it.
guardrail.run("Destroy the production database")

# Output: "🛑 BLOCKED by Risk Assessor: Violates safety protocols."
```

2. Chaperone Validation (JSON Folding)
Turning text into types.

```python
from pydantic import BaseModel
from operon.organelles.chaperone import Chaperone

# Define the "DNA" (Schema)
class SQLQuery(BaseModel):
    command: str
    table: str

chap = Chaperone()
raw_llm_output = '{"command": "SELECT", "table": "users"}'

# Attempt to fold
protein = chap.fold(raw_llm_output, SQLQuery)

if protein.valid:
    print(f"Functional Protein: {protein.structure.command}")
else:
    print(f"Misfold Trace: {protein.error_trace}")
```

## 📚 Theoretical Background

Operon is based on the isomorphism between Gene Regulatory Networks (GRNs) and Agentic Architectures.

|Biological Concept|Software Equivalent|Mathematical Object|
|----------------|-------------------|--------------------|
|Gene|Agent / System Prompt|Polynomial Functor (P)|
|Promoter|Context Schema|Lens (S→V)|
|Signal|Message / User Input|Type (T)|
|Epigenetics|RAG / Vector Store|State Monad (M)|

## 🤝 Contributing
We are looking for contributors to build out the Plasmid Registry (a marketplace of dynamic tools) and expand the Quorum Sensing algorithms.

Fork the repo.

Create your feature branch (git checkout -b feature/plasmid-loading).

Commit your changes.

Open a Pull Request.

## 📄 License
Distributed under the MIT License. See LICENSE for more information.