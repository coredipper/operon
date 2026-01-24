# Example 39: Chaperone Healing Loop

Demonstrates the biological Chaperone Loop pattern where validation failures
are fed back to the generator for context-aware refolding.

```mermaid
flowchart LR
    subgraph ChaperoneLoop["Chaperone Loop (GroEL/GroES)"]
        prompt[Prompt] --> generator[Generator LLM]
        generator --> raw[Raw Output]
        raw --> chaperone[Chaperone Validator]
        chaperone -->|valid| output[Folded Protein]
        chaperone -->|invalid| error[Error Trace]
        error -->|feedback| generator
        error -->|max retries| ubiquitin[Ubiquitin Tag]
        ubiquitin --> lysosome[Lysosome]
    end

    style output fill:#c8e6c9
    style ubiquitin fill:#ffcdd2
    style lysosome fill:#ffcdd2
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant P as Prompt
    participant G as Generator
    participant C as Chaperone
    participant O as Output
    participant L as Lysosome

    P->>G: Generate output
    G->>C: Raw JSON
    alt Valid
        C->>O: Folded Protein ✓
    else Invalid (retry < max)
        C->>G: Error trace (feedback)
        G->>C: Corrected JSON
        C->>O: Healed output
    else Invalid (max retries)
        C->>L: Ubiquitin tagged
    end
```

## ASCII Wiring

```
                    +---------------------------+
                    |                           |
                    v                           |
[prompt] --text(U)--> [generator] --json(U)--> [chaperone] --json(V)--> [output]
                           ^                        |
                           |                        |
                           +---error(V)---[healing_feedback]
                                                    |
                                          [max_retries?]--ubiquitin(V)--> [lysosome]

Confidence: 1.0 → 0.85 → 0.70 → ... (decay per retry)
```

## Key Concepts

- **Feedback-driven repair**: Error traces guide regeneration
- **Confidence decay**: Each retry reduces output confidence (0.15 per attempt)
- **Ubiquitin tagging**: Mark unfixable outputs for degradation

## Biological Parallel

- **GroEL/GroES**: Isolation chamber giving proteins a second chance to fold
- **Unfolded Protein Response**: Stress pathway when folding repeatedly fails

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
