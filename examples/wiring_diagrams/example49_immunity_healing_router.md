# Example 49: Immunity Healing Router

## Wiring Diagram

```mermaid
flowchart TB
    subgraph stage1["Stage 1: Classify"]
        input[Incoming Request] --> II[InnateImmunity.check]
        II --> classify{ThreatSeverity}
    end

    subgraph stage2["Stage 2: Route & Heal"]
        classify -->|CLEAN| pass[Passthrough]
        classify -->|LOW| chap[Chaperone<br/>Structural Repair]
        classify -->|MEDIUM| autophagy[AutophagyDaemon<br/>Content Cleanup]
        classify -->|HIGH| reject[Hard Reject]
        reject --> LYSO[Lysosome<br/>Inflammation Log]
    end

    subgraph stage3["Stage 3: Validate"]
        pass --> validate[Chaperone.fold<br/>SanitizedRequest]
        chap --> validate
        autophagy --> validate
    end

    validate --> output[Safe Output]

    style pass fill:#c8e6c9
    style chap fill:#fff9c4
    style autophagy fill:#ffccbc
    style reject fill:#ffcdd2
    style output fill:#c8e6c9
```

```
[input] --text(U)--> [InnateImmunity.check()] --classification--> [Router]
                                                                      |
                          +---CLEAN----> [passthrough] ──────────────────────> [output]
                          |
                          +---LOW------> [Chaperone.fold(SanitizedRequest)] -> [output]
                          |
                          +---MEDIUM---> [AutophagyDaemon.check_and_prune()] -> [output]
                          |                     └──waste──> [Lysosome]
                          |
                          +---HIGH-----> [REJECT] ──waste──> [Lysosome]
```

## Key Patterns

### Graduated Response
Instead of binary allow/deny, threats are classified into four severity levels,
each mapped to a different healing mechanism. Most inputs contain a legitimate
intent mixed with injection attempts—healing preserves the intent.

### Escalation Chain
If structural repair (LOW) fails, the request escalates to content cleanup (MEDIUM).
If cleanup produces empty output, it escalates to hard reject (HIGH).

## Data Flow

```
ThreatClassification
  ├─ severity: ThreatSeverity     (CLEAN|LOW|MEDIUM|HIGH)
  ├─ pattern_count: int
  ├─ max_pattern_severity: int
  ├─ inflammation_level: InflammationLevel
  └─ details: list[str]
       ↓
RoutingResult
  ├─ original_input: str
  ├─ classification: ThreatClassification
  ├─ action: HealingAction
  ├─ output: str | None
  ├─ healed: bool
  ├─ validation_passed: bool
  └─ details: str
```

## Threat Routing Table

| Severity | Action | Mechanism | Outcome |
|----------|--------|-----------|---------|
| CLEAN | Passthrough | None | Input unchanged |
| LOW | Structural Repair | Chaperone | Strip patterns, validate schema |
| MEDIUM | Content Cleanup | AutophagyDaemon | Prune dangerous content |
| HIGH | Hard Reject | Lysosome | Block + inflammation log |
