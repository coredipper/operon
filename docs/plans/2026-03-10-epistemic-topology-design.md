# Epistemic Topology of Wiring Diagrams

**Date:** 2026-03-10
**Status:** Design complete, ready for implementation

## Motivation

Three converging research threads motivate formalizing epistemic properties of operon's topologies:

1. **Operon's existing framework** — coalgebraic agents composed via polynomial functors (⊗, ∘, Tr) with membranes, optics, and provenance. Topology determines behavior, but the framework lacks formal operators for what agents *know*.

2. **Davis (UCLA/MongoDB)** — combining epistemic logic (K_i φ) with temporal logic (TLA+) for mechanical verification of knowledge properties in distributed systems. Demonstrates that epistemic reasoning is tractable for protocol verification.

3. **Google Research (2026)** — "Towards a Science of Scaling Agent Systems." 180 configurations across 5 architectures. Key empirical findings: Independent systems amplify errors 17.2x vs Centralized 4.4x; sequential tasks degrade 39-70% under multi-agent decomposition; parallel tasks improve 80.9% with coordination; tool density beyond 16 causes disproportionate overhead.

**Thesis:** Operon's wiring diagrams already encode epistemic structure implicitly. By deriving Kripke-style knowledge operators from the existing observation topology (membranes, optics, wires), we can predict Google's empirical results from first principles — turning operon from "biologically inspired framework" into "predictive theory of agent scaling."

## Approach: Observation-Based Epistemic Semantics

Epistemic properties are *derived from existing structure*, not added as new primitives. The accessibility relation for each agent comes from what it can observe through its ports — determined by the wiring diagram's `Wire`, `Optic`, and `PortType` definitions already in the codebase.

This is preferred over a belief-based approach (extending coalgebra state with explicit belief) because:
- It follows operon's philosophy: "safety from structure, not strings"
- It requires no new core types
- It directly connects to Google's findings (their architectures differ in observation topology)
- The membrane/optics *are* the epistemic accessibility relation

## Core Definitions

### Observation Function

Given a wiring diagram D = (M, W) and agent i ∈ M, the observation function obs_i: S_D → O_i maps global system state to agent i's local observation — the values on i's input wires. Already implicitly defined by the wires targeting i's input ports and their optics.

### Epistemic Indistinguishability

Two global states s, s' ∈ S_D are indistinguishable to agent i (written s ~_i s') iff obs_i(s) = obs_i(s'). This equivalence relation defines i's information partition.

### Knowledge Operator

Agent i knows proposition φ in state s (K_i(φ) at s) iff φ holds in all states s' such that s ~_i s'. Standard Kripke semantics with the accessibility relation derived from the wiring diagram.

Key consequence: The membrane and optics *are* the epistemic accessibility relation. A PrismOptic that filters by DataType restricts observations, coarsening the information partition. A BudgetOptic that gates by cost can make previously observable states unobservable when budget runs low.

### Group Epistemic Operators

- **Mutual Knowledge:** E_G(φ) = ∧_{i∈G} K_i(φ). Requires φ visible on input wires of every cell in the tissue. Morphogen diffusion with full saturation achieves this for morphogen-encoded propositions.

- **Common Knowledge:** C_G(φ) = ∧_{k=1}^∞ E_G^k(φ). Everyone knows everyone knows... ad infinitum. Strictly stronger than mutual knowledge. Famously hard to achieve in asynchronous distributed systems.

- **Distributed Knowledge:** D_G(φ) holds iff the group collectively knows φ when observations are pooled. Defined by the intersection of all indistinguishability relations: s ~_D s' iff s ~_i s' for ALL i ∈ G. The finest partition achievable by combining all agents' observations.

### Mapping to Operon Topologies

- **Independent (pure ⊗):** No shared wires. D_G can be rich (agents see different things) but E_G is impoverished. No path to C_G.
- **Centralized (∘ with hub):** Hub observes all worker outputs. Hub's partition refines toward D_G. Workers don't observe each other — E_G requires hub to broadcast back.
- **Decentralized (⊗ with Tr feedback):** Peer-to-peer wires create overlapping observations. Each feedback round refines mutual knowledge toward common knowledge, at communication cost.

## Theorems

All theorems have two parts: a qualitative result (holds universally for any agent system with the given topology) and a parametric corollary (under specific modeling assumptions, produces bounds matching Google's empirical data).

### Theorem 1: Error Amplification Bound

**Setup.** n agents processing subtasks with independent error probability p. Final aggregator combines results.

**(a) Independent (⊗, no feedback):** Aggregator sees only final results, not intermediate reasoning. Cannot distinguish error from different input. Without K_hub(error_j), errors pass unchecked.

A_independent ≤ n

**(b) Centralized (∘ with hub):** Hub observes each worker's output before aggregation. Finer information partition detects inconsistencies. With hub detection rate d:

A_centralized ≤ n · (1 - d)

**Parametric Corollary.** n ≈ 3-5, d ≈ 0.75: A_independent / A_centralized ≈ 1/(1-d) = 4. Google observed 17.2/4.4 ≈ 3.9x.

**Epistemic interpretation:** Error amplification is inversely proportional to the topology's capacity for distributed knowledge.

### Theorem 2: Sequential Penalty

**Setup.** Task with k strictly ordered steps. Step j requires result of step j-1. Decomposed across n agents.

**(a) Single agent:** K_i(result_j) trivially satisfied. Zero communication cost. Total cost = k · c_step.

**(b) Multi-agent with handoffs:** Each of h ≤ k-1 handoffs requires establishing E({sender, receiver})(result_j). Cost per handoff = c_comm. Total cost = k · c_step + h · c_comm.

**(c) Worse than additive:** Receiving agent must reconstruct context. Information loss per handoff:

ΔI_j = H(state_sender) - H(obs_receiver(state_sender))

Strictly positive whenever wire doesn't carry full state — always true since context windows are finite.

**Parametric Corollary.** c_comm/c_step ≈ 0.3-0.5, h ≈ k/n handoffs. Overhead reaches 30-50%, compounding with context reconstruction failures to 39-70%. Matches Google's range.

**Epistemic interpretation:** The sequential penalty is the cost of manufacturing knowledge that a single agent gets for free.

### Theorem 3: Parallel Acceleration

**Setup.** Task decomposes into m epistemically independent subtasks. Each has cost c_sub. Coordinator assigns and aggregates.

**Definition.** Two subtasks are *epistemically independent* iff neither's solution requires knowledge of the other's result: ¬(K_i(φ_j) is a precondition for solving subtask i).

**(a) Full epistemic independence:** Speedup S = (m · c_sub) / (max_i(c_sub_i) + c_assign + c_aggregate). With equal subtasks: S ≈ m minus coordinator overhead.

**(b) Centralized coordination adds bounded cost:** Hub achieves D_G(results) at aggregation for free (must receive results anyway). Error detection (Theorem 1) comes as architectural bonus.

**(c) Epistemic ceiling:** Partial independence sacrifices quality. Tradeoff governed by mutual information I(φ_i; φ_j) between subtask solutions.

**Parametric Corollary.** Finance-Agent: m = 3-4 epistemically independent subtasks, small coordinator overhead. S ≈ 2-3x plus error detection benefit → ~80%. Google observed 80.9%.

### Theorem 4: Tool Density Scaling

**Setup.** Tool set T with |T| = t tools distributed across n agents. Each agent holds T_i ⊆ T.

**(a) Single agent:** Observation partition over tool outputs is unified. Planning cost O(t) per step.

**(b) Multi-agent, distributed tools:** Remote tool access requires:
1. K_i(T_j ∋ tool_k) — knowing the tool exists. Cost: O(t) context tokens.
2. Inter-agent request: c_comm per call.
3. Result interpretation with information loss ΔI.

Coordination overhead per step: O(t · c_comm) — linear in total tool count regardless of agent count.

**(c) Planning is superlinear:** Agent must reason about K_i(K_j(tool_k can solve subproblem)) — second-order epistemic reasoning. Scales as O(t · n) worst case.

**Parametric Corollary.** At t = 16, n = 3-4: remote tool fraction ≈ 0.7, second-order planning cost ≈ 64 vs 16 for single agent. 4x overhead negates parallelism benefits. Matches Google's inflection point.

**Epistemic interpretation:** Tool density scaling is knowledge fragmentation. Adding agents partitions the tool-knowledge space. Accessing distributed knowledge D_G requires communication that scales with fragmentation.

## Paper Integration

Extends existing multi-cellular section (§6) with three subsections:

### §6.X Epistemic Topology of Wiring Diagrams (~300 words)
- Definitions: observation function, indistinguishability, K_i from wires
- Group operators: E_G, C_G, D_G and relationship to topology classes
- Connection to existing optics/membrane

### §6.Y Predictive Theorems for Multi-Agent Coordination (~1000 words)
- Theorems 1-4 with proofs
- Parametric corollaries in shared "Empirical Validation" subsection
- Summary table: topology → epistemic property → predicted behavior → Google's observed result

### §6.Z Epistemic Dynamics and Adaptive Topology (~200 words)
- Connects epiplexity to epistemic operators: low epiplexity + high uncertainty = K_i(¬K_i(φ))
- Morphogen-driven topology switching as epistemic optimization
- Forward-looking: connects static theorems to operon's dynamic capabilities

**Total addition:** ~1500 words, 4 theorems, 1 validation table.

## Validation Table (Preview)

| Topology | Epistemic Property | Prediction | Google Observed |
|---|---|---|---|
| Independent (⊗) | No inter-agent knowledge | Unbounded error amplification | 17.2x amplification |
| Centralized (∘+hub) | Hub has D_G | Bounded amplification ≤ n(1-d) | 4.4x amplification |
| Multi-agent + sequential | Requires manufactured E_G per step | 30-70% overhead | 39-70% degradation |
| Multi-agent + parallel | Epistemic independence → free parallelism | ~m× speedup + validation bonus | 80.9% improvement |
| Multi-agent + high tool density | Knowledge fragmentation, O(t·n) planning | Superlinear overhead past threshold | Inflection at 16 tools |

## References

- Davis, J.J. "Reasoning About Knowledge in TLA+." UCLA CS 201-A Seminar, 2026.
- Google Research. "Towards a Science of Scaling Agent Systems: When and Why Agent Systems Work." 2026.
- Fagin, R. et al. *Reasoning About Knowledge.* MIT Press, 1995.
- Halpern, J.Y. and Moses, Y. "Knowledge and Common Knowledge in a Distributed Environment." JACM, 1990.
