# Example 96: Co-Design Adapter Composition

## Wiring Diagram

```
DesignProblems
  ├── SwarmTopologyAdapter (pattern → risk + advice)
  └── MemoryBridgeAdapter  (entries → facts + audit)
        |           |
        v           v
  [compose_series]    [compose_parallel]
    A then B            A and B on same input
        |                     |
        v                     v
  composite DP           composite DP
        |                     |
        └─────────┬───────────┘
                  v
        [feasibility_check]
          ├─ feasible: bool
          └─ functionalities: dict
                  |
                  v
        [feedback_fixed_point]
          DP + initial state
                  |
          ┌───── loop ─────┐
          │  evaluate_fn    │
          │  check delta    │
          │  < epsilon?     │
          └─────────────────┘
                  |
                  v
          converged state
            (score, iterations)
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | DesignProblem | Unit of co-design: evaluate_fn + feasibility_fn |
| 2 | compose_series() | Chain A's output as B's input |
| 3 | compose_parallel() | Run A and B on same input, merge outputs |
| 4 | feasibility_check() | Tests whether input satisfies feasibility_fn |
| 5 | feedback_fixed_point() | Iterates DP until convergence_key stabilizes |
| 6 | epsilon + max_iterations | Convergence control for fixed-point loop |
| 7 | Zardini co-design | Formal composition algebra over adapter problems |

## Data Flow

```
Individual DPs:
  SwarmTopologyAdapter:
    input:  {pattern: "HierarchicalSwarm", agents: 4}
    output: {topology_advice: "analyzed_...", risk_score: 0.3}

  MemoryBridgeAdapter:
    input:  {entries: [{id: 1}, {id: 2}]}
    output: {facts_created: 2, audit_trail: true}
       ↓
compose_series(swarms_dp, memory_dp)
  "SwarmsThenMemory"
  └─ Not feasible: swarms output lacks "entries" key for memory input
       ↓
compose_parallel(swarms_dp, memory_dp)
  "SwarmsAndMemory"
  input: {pattern: "HierarchicalSwarm", agents: 3, entries: [{id: 1}]}
  └─ Feasible: both DPs satisfied, outputs merged
       ↓
feedback_fixed_point(scoring_dp, initial={score: 0.5}, epsilon=0.001)
  iteration 1: score = 0.5 * 0.8 + 0.95 * 0.2 = 0.59
  iteration 2: score = 0.59 * 0.8 + 0.95 * 0.2 = 0.662
  ...
  iteration N: score -> 0.9500 (converged)
  └─ {converged: true, iterations: N, final_score: 0.9500}
```

## Composition Algebra

| Operation | Inputs | Output | Feasibility |
|-----------|--------|--------|-------------|
| Series (A;B) | A then B | B(A(input)) | A feasible AND B feasible on A's output |
| Parallel (A\|B) | Same input | merge(A(input), B(input)) | Both feasible on input |
| Feedback | DP + initial | Fixed point | Converges within max_iterations |
