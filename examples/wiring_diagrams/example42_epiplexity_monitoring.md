# Example 42: Epiplexity Monitoring

Detects epistemic stagnation in agents through Bayesian Surprise monitoring.
Based on the Free Energy Principle: healthy agents minimize surprise through
learning or effective action; stagnant agents do neither.

```mermaid
flowchart TB
    subgraph Monitor["Epiplexity Monitor"]
        input[Agent Output] --> embed[Embedding Provider]
        embed --> novelty["Embedding Novelty (ê)<br/>½(1 - cos(e_t, e_{t-1}))"]
        input --> perplexity["Perplexity (H)<br/>σ(H) = 1 - e^{-H/H₀}"]
        novelty --> combine["Ê = α·ê + (1-α)·σ(H)"]
        perplexity --> combine
        combine --> window[Windowed Integral E_w]
        window --> status{Health Status}
        status -->|high novelty| exploring[EXPLORING]
        status -->|E_w above δ| healthy[HEALTHY]
        status -->|low novelty, low H| converging[CONVERGING]
        status -->|E_w below δ| stagnant[STAGNANT]
        status -->|sustained E_w below δ| critical[CRITICAL]
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
    HEALTHY --> EXPLORING: High novelty
    EXPLORING --> HEALTHY: Moderate Ê
    HEALTHY --> CONVERGING: Low novelty + low perplexity
    CONVERGING --> HEALTHY: Ê recovers
    HEALTHY --> STAGNANT: E_w drops below δ
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
                                               (Ê above δ)          (Ê below δ)          (sustained)
```

## Key Insight

If an agent's outputs stabilize (low embedding novelty ê) while its perplexity
remains high (model is uncertain H), it's in a pathological loop—not converging
to a solution. Low Epiplexity = low Bayesian surprise = stagnation.

## Formula

```
Ê_t = α·½(1 - cos(e_t, e_{t-1})) + (1-α)·σ(H(m_t|m_{<t}))
```

Where:
- `e_t`: Embedding of message at time t
- `H`: Perplexity (model uncertainty)
- `α`: Balance parameter (default 0.5)
- `σ(H)`: Exponential saturation (1 - e^{-H/H₀})

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
