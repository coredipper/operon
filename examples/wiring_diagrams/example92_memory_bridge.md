# Example 92: Memory Bridge

## Wiring Diagram

```
AnimaWorks entries             DeerFlow session + vectors
  [{id, type, content,           session: [{role, content, timestamp}]
    timestamp, source_agent}]    vectors: [{id, content, metadata}]
        |                                |
        v                                v
  [bridge_animaworks_memory]   [bridge_deerflow_memory]
        |                                |
        └────────────┬───────────────────┘
                     v
             BiTemporalMemory
               (unified store)
                     |
                     v
           [retrieve_known_at]
                     |
                     v
             Fact list (auditable)
               [{source, subject, value}]
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | bridge_animaworks_memory() | Converts AnimaWorks episodic/semantic entries to facts |
| 2 | bridge_deerflow_memory() | Converts session turns + vector docs to facts |
| 3 | BiTemporalMemory | Unified bi-temporal store with audit trail |
| 4 | retrieve_known_at() | Point-in-time query across all bridged sources |
| 5 | Fact provenance | Each fact tracks original source framework |
| 6 | session_id parameter | Groups DeerFlow facts by session for traceability |

## Data Flow

```
AnimaWorks entries:
  ├─ {type: "episodic", content: "User prefers Python"}
  └─ {type: "semantic", content: "Project uses FastAPI"}
       ↓
bridge_animaworks_memory(entries, mem)
  └─ 2 facts recorded
       ↓
DeerFlow session + vectors:
  ├─ session: [{role: "user", content: "Find AI papers"}, ...]
  └─ vectors: [{content: "Scaling laws for neural LMs", ...}]
       ↓
bridge_deerflow_memory(session, vectors, mem, session_id="demo_session_1")
  └─ 3 facts recorded (2 session turns + 1 vector doc)
       ↓
BiTemporalMemory
  └─ 5 total facts
       ↓
mem.retrieve_known_at(at=timestamp)
  └─ [{source: "animaworks", subject: ..., value: ...},
      {source: "deerflow",   subject: ..., value: ...}, ...]
```

## Bridged Sources

| Source | Input Format | Facts Created | Fact Source Tag |
|--------|-------------|---------------|---------------|
| AnimaWorks | episodic/semantic entries | 1 per entry | "animaworks" |
| DeerFlow | session turns + vector docs | 1 per turn + 1 per vector | "deerflow" |
