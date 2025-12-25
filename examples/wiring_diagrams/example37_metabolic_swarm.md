# Example 37: Metabolic Swarm Budgeting

This diagram illustrates the Metabolic Coalgebra formalism where agents share
a finite token budget. The structure map alpha: S -> P(S) + bot is partial,
returning bot (apoptosis) when resources are insufficient.

```mermaid
flowchart TB
  subgraph SharedMitochondria["SharedMitochondria (ATP Pool)"]
    budget[ATP Budget]
  end

  subgraph Swarm["MetabolicSwarm (Tensor Product)"]
    task[Task Input] -->|"task (U)"| fanout[Fanout]

    fanout -->|"task (U)"| worker1[Worker_1]
    fanout -->|"task (U)"| worker2[Worker_2]
    fanout -->|"task (U)"| worker3[Worker_3]

    worker1 -->|"result (V)"| collector[Candidate Collector]
    worker2 -->|"result (V)"| collector
    worker3 -->|"result (V)"| collector

    collector -->|"candidates (V)"| verifier[Verifier]
    verifier -->|"verdict (T)"| output[Result Output]
  end

  budget -.->|"consume(c)"| worker1
  budget -.->|"consume(c)"| worker2
  budget -.->|"consume(c)"| worker3
  budget -.->|"consume(c)"| verifier

  worker1 -.->|"r < c"| apoptosis1[DEAD]
  worker2 -.->|"r < c"| apoptosis2[DEAD]
  worker3 -.->|"r < c"| apoptosis3[DEAD]
  verifier -.->|"r < c"| apoptosis_v[DEAD]

  style budget fill:#f9f,stroke:#333
  style apoptosis1 fill:#f66,stroke:#333
  style apoptosis2 fill:#f66,stroke:#333
  style apoptosis3 fill:#f66,stroke:#333
  style apoptosis_v fill:#f66,stroke:#333
```

## Coalgebraic Interpretation

```
State S = L × R  (Logical State × Resource State)

Transition: alpha(l, r) = | (l', r - c)  if r >= c
                          | bot          if r < c  (Apoptosis)

Halting Guarantee: R is strictly decreasing, bounded below by 0
```

## Termination Conditions

| Condition | Cause | Description |
|-----------|-------|-------------|
| Solution verified | `solved` | Worker found solution, verifier confirmed |
| Budget exhausted | `ischemia` | Global ATP reached 0 |
| All workers dead | `swarm_collapse` | All workers starved before solution |
| Verifier dead | `verifier_death` | Verifier starved during verification |
| Cycle limit | `entropy_limit` | Max iterations reached (entropy check) |

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED, c = cost per step, r = remaining budget.
