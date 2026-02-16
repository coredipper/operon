# Example 46: Codebase Q&A with RAG

## Wiring Diagram

```mermaid
flowchart TB
    subgraph input["Input"]
        Q[("User Question")]
    end

    subgraph stage1["Stage 1: Defense"]
        Q --> Membrane["Membrane<br/>━━━━━━━━━━━━━<br/>• Injection detection<br/>• Jailbreak patterns"]
    end

    subgraph stage2["Stage 2: Budget"]
        Membrane --> ATP["ATP_Store<br/>━━━━━━━━━━━━━<br/>• 100 ATP per query<br/>• Reject if exhausted"]
    end

    subgraph stage3["Stage 3: Planning"]
        ATP --> Planner["QueryPlanner<br/>━━━━━━━━━━━━━<br/>• Pattern matching<br/>• Memory lookup<br/>• Keyword extraction"]
    end

    subgraph stage4["Stage 4: Search"]
        Planner --> Searcher["CodeSearcher<br/>━━━━━━━━━━━━━<br/>• Grep patterns<br/>• Glob matching<br/>• Definition search"]
    end

    subgraph stage5["Stage 5: Context"]
        Searcher --> Assembler["ContextAssembler<br/>━━━━━━━━━━━━━<br/>• Limit snippets<br/>• Format results"]
    end

    subgraph stage6["Stage 6: Generate + Heal"]
        Assembler --> Generator["Generator<br/>━━━━━━━━━━━━━<br/>• LLM or Mock<br/>• Citation format"]
        Generator --> Chaperone["Chaperone<br/>━━━━━━━━━━━━━<br/>• Schema validation"]
        Chaperone --> Validator["CitationValidator<br/>━━━━━━━━━━━━━<br/>• File exists?<br/>• Line valid?"]
        Validator -->|"Invalid"| Generator
    end

    subgraph stage7["Stage 7: Memory"]
        Validator -->|"Valid"| Memory["HistoneStore<br/>━━━━━━━━━━━━━<br/>• Store patterns<br/>• Acetylation markers"]
    end

    subgraph output["Output"]
        Memory --> Answer[("CodeAnswer<br/>answer + citations")]
    end

    style Q fill:#e1f5fe
    style Answer fill:#c8e6c9
    style Validator fill:#fff9c4
```

## Key Patterns

### Chaperone Healing Loop
If the generator produces hallucinated file citations (e.g., `nonexistent.py:42`),
the validator rejects and retries with error feedback. This is the GroEL/GroES
pattern - giving outputs a chance to refold correctly.

### Histone Memory
Successful search patterns are stored as acetylation markers with 1-week decay.
Future queries check memory first, boosting patterns that worked before.

### Query Planning
Detects question types:
- "Where is X defined?" → definition search
- "What files contain X?" → grep search
- Generic → keyword extraction

## Data Flow

```
SearchResult
  ├─ file_path: str
  ├─ line_number: int
  └─ line_content: str
       ↓
CodeAnswer
  ├─ answer: str
  ├─ citations: list[Citation]
  ├─ confidence: float
  └─ search_patterns_used: list[str]
       ↓
QueryResult
  ├─ success: bool
  ├─ answer: CodeAnswer
  ├─ healing_attempts: int
  └─ search_results_count: int
```
