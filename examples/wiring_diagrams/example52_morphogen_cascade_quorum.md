# Example 52: Morphogen Cascade with Quorum

## Wiring Diagram

```mermaid
flowchart TB
    subgraph input["Document Input"]
        doc[Contract/Document] --> meta[Extract Metadata]
        meta --> MG[MorphogenGradient<br/>Initialize from metadata]
    end

    subgraph stage1["Stage 1: Extract Clauses"]
        MG --> extract[Parse Document]
        extract --> clauses[Clause List]
        clauses --> update1["Update COMPLEXITY<br/>morphogen"]
        update1 --> MG
    end

    subgraph stage2["Stage 2: Risk Assessment"]
        MG --> check{Confidence Level}
        check -->|HIGH| single[Single Reviewer]
        check -->|LOW| quorum[QuorumSensing<br/>3 Voters]
        single --> risk[Risk Score]
        quorum --> risk
        risk --> update2["Update RISK<br/>morphogen"]
        update2 --> MG
    end

    subgraph stage3["Stage 3: Final Decision"]
        MG --> threshold[Read RISK morphogen]
        threshold --> verdict{Approval?}
        verdict -->|yes| approve[APPROVED]
        verdict -->|no| deny[DENIED]
    end

    subgraph feedback["Feedback"]
        NFL[NegativeFeedbackLoop]
        verdict --> NFL
        NFL --> adjust["Adjust confidence<br/>threshold"]
        adjust --> MG
    end

    style approve fill:#c8e6c9
    style deny fill:#ffcdd2
    style quorum fill:#e3f2fd
    style MG fill:#e0f7fa
```

```
[Document] --metadata--> [MorphogenGradient]
                               |
[Stage 1: Extract Clauses] <---+---> update COMPLEXITY
                               |
[Stage 2: Risk Assessment] <---+
    ├── confidence HIGH --> [Single Reviewer] --risk--> update RISK
    └── confidence LOW  --> [QuorumSensing(3)] --risk--> update RISK
                               |
[Stage 3: Decision] <---------+---> read RISK --> verdict
                               |
[NegativeFeedbackLoop] <-------+---> adjust confidence threshold
```

## Key Patterns

### Dynamic Quorum Activation
Quorum voting is not always-on — it activates only when the morphogen gradient
indicates low confidence. Simple documents get a single reviewer; complex or
ambiguous ones trigger multi-voter consensus.

### Morphogen-Linked Cascade
Each cascade stage both reads and writes morphogen signals. Stage 1 sets
COMPLEXITY, Stage 2 reads CONFIDENCE and writes RISK, Stage 3 reads RISK
for the final threshold comparison.

### Feedback-Adjusted Confidence
The NegativeFeedbackLoop adjusts confidence thresholds based on voting outcomes.
If quorum frequently disagrees, confidence drops (more quorum activations).

## Data Flow

```
ComplianceDocument
  ├─ title: str
  ├─ content: str
  ├─ complexity: float
  └─ risk_level: float
       ↓
ClauseExtraction
  ├─ clauses: list[str]
  ├─ clause_count: int
  └─ complexity_score: float
       ↓
RiskAssessment
  ├─ risk_score: float
  ├─ quorum_used: bool
  ├─ voter_count: int
  └─ consensus: float
       ↓
ReviewVerdict
  ├─ approved: bool
  ├─ risk_score: float
  ├─ confidence: float
  └─ reasoning: str
```

## Morphogen Signals

| Morphogen | Set By | Read By | Range |
|-----------|--------|---------|-------|
| COMPLEXITY | Stage 1 | Stage 2 | 0.0 - 1.0 |
| CONFIDENCE | Feedback Loop | Stage 2 | 0.0 - 1.0 |
| RISK | Stage 2 | Stage 3 | 0.0 - 1.0 |
| ERROR_RATE | Feedback Loop | Stage 2 | 0.0 - 1.0 |
