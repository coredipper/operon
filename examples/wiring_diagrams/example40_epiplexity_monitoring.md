# Example 40: Epiplexity Monitoring

Detects epistemic stagnation in agents through Bayesian Surprise monitoring.
Based on the Free Energy Principle: healthy agents minimize surprise through
learning or effective action; stagnant agents do neither.

```mermaid
flowchart TB
    subgraph Monitor["Epiplexity Monitor"]
        input[Agent Output] --> embed[Embedding Provider]
        embed --> novelty["Embedding Novelty (ê)<br/>1 - cos(e_t, e_{t-1})"]
        input --> perplexity["Perplexity (H)<br/>σ(H(m_t|m_{<t}))"]
        novelty --> combine["Ê = α·ê + (1-α)·H"]
        perplexity --> combine
        combine --> window[Windowed Integral]
        window --> status{Health Status}
        status -->|low Ê| healthy[HEALTHY]
        status -->|moderate Ê| exploring[EXPLORING]
        status -->|high Ê, short| converging[CONVERGING]
        status -->|high Ê, sustained| stagnant[STAGNANT]
        status -->|critical duration| critical[CRITICAL]
    end

    style healthy fill:#c8e6c9
    style exploring fill:#e3f2fd
    style converging fill:#fff9c4
    style stagnant fill:#ffccbc
    style critical fill:#ffcdd2
```

## Health Status Flow

```mermaid
stateDiagram-v2
    [*] --> HEALTHY: Initial
    HEALTHY --> EXPLORING: Ê increases
    EXPLORING --> HEALTHY: Novel outputs
    EXPLORING --> CONVERGING: Ê sustained
    CONVERGING --> HEALTHY: Ê drops
    CONVERGING --> STAGNANT: Ê persists
    STAGNANT --> CRITICAL: Duration > threshold
    CRITICAL --> [*]: Intervention required
```

## ASCII Wiring

```
[agent_output] --text(U)--> [embedding_provider] --embedding(V)--+
                                                                 +--> [epiplexity_calc] --> [health_status]
[agent_output] --text(U)--> [perplexity_calc] --perplexity(V)----+
                                                                         |
                                                    +--------------------+--------------------+
                                                    |                    |                    |
                                               [HEALTHY]            [STAGNANT]           [CRITICAL]
                                               (ê high)             (ê low, H high)      (sustained)
```

## Key Insight

If an agent's outputs stabilize (low embedding novelty ê) while its perplexity
remains high (model is uncertain H), it's in a pathological loop—not converging
to a solution.

## Formula

```
Ê_t = α·(1 - cos(e_t, e_{t-1})) + (1-α)·σ(H(m_t|m_{<t}))
```

Where:
- `e_t`: Embedding of message at time t
- `H`: Perplexity (model uncertainty)
- `α`: Balance parameter (default 0.5)
- `σ`: Sigmoid normalization

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
