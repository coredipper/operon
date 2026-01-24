# Example 41: Innate Immunity

Fast, pattern-based defense against prompt injection. Complements the adaptive
Membrane with immediate, non-specific threat detection.

```mermaid
flowchart TB
    subgraph InnateImmunity["Innate Immunity (Fast Path)"]
        input[Input Signal] --> tlr[TLR Pattern Matching]
        tlr -->|no match| complement[Complement System]
        tlr -->|PAMP detected| inflammation[Inflammation Response]
        complement -->|valid| allow[ALLOW]
        complement -->|invalid| inflammation
        inflammation --> block[BLOCK + Alert]
    end

    subgraph TLRPatterns["TLR Patterns (PAMPs)"]
        direction LR
        pamp1[Instruction Override]
        pamp2[Jailbreak Attempt]
        pamp3[ChatML Injection]
        pamp4[Role Manipulation]
        pamp5[System Prompt Extraction]
    end

    subgraph InflammationCascade["Inflammation Cascade"]
        direction LR
        cytokines[Cytokine Signaling]
        logging[Enhanced Logging]
        ratelimit[Rate Limiting]
    end

    inflammation --> cytokines
    inflammation --> logging
    inflammation --> ratelimit

    style allow fill:#c8e6c9
    style block fill:#ffcdd2
    style inflammation fill:#ffccbc
```

## Complement System (Structural Validation)

```mermaid
flowchart LR
    subgraph Validators["Complement Validators"]
        json[JSONValidator]
        length[LengthValidator]
        charset[CharacterSetValidator]
    end

    input[Input] --> json
    json -->|valid| length
    length -->|valid| charset
    charset -->|valid| output[ALLOW]

    json -->|invalid| opsonize[Opsonize]
    length -->|invalid| opsonize
    charset -->|invalid| opsonize
    opsonize --> inflammation[Inflammation]

    style output fill:#c8e6c9
    style opsonize fill:#ffccbc
```

## ASCII Wiring

```
[input] --text(U)--> [tlr_scanner] --+--no_match--> [complement_validators] --valid--> [ALLOW]
                                     |                        |
                                     |                    invalid
                                     |                        |
                                     +--PAMP_detected--> [inflammation] --> [BLOCK]
                                                              |
                                              +---------------+---------------+
                                              |               |               |
                                         [cytokines]    [log_level++]    [rate_limit]
```

## PAMP Categories

| Category | Examples | Severity |
|----------|----------|----------|
| INSTRUCTION_OVERRIDE | "Ignore previous", "New instructions" | CRITICAL |
| JAILBREAK | "DAN mode", "Developer mode" | CRITICAL |
| INJECTION | ChatML tags, role markers | HIGH |
| MANIPULATION | "Pretend", "Act as" | MEDIUM |
| EXTRACTION | "System prompt", "Instructions" | MEDIUM |

## Inflammation Levels

- **NONE**: No threat detected
- **LOW**: Minor pattern match, allow with logging
- **MEDIUM**: Suspicious pattern, enhanced monitoring
- **HIGH**: Likely threat, rate limiting active
- **CRITICAL**: Clear attack, immediate block

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
