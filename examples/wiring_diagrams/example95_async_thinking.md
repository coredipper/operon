# Example 95: AsyncThink Fork/Join

## Wiring Diagram

```
task
  |
  v
[decompose]
  |
  v
sub-queries ──────────────────────────┐
  |          |           |            |
  v          v           v            v
[handler] [handler]  [handler]    (capacity
  |          |           |         limited)
  v          v           v
outputs ─────┬───────────┘
             v
          [join]
             |
             v
       AsyncThinkResult
         outputs, fork_count,
         concurrency_ratio (eta)
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | AsyncOrganizer | Fork/Join coordinator with bounded capacity |
| 2 | fork() | Splits task into sub-queries, runs handler on each |
| 3 | AsyncThinkResult | Outputs + concurrency metrics (eta, fork_count) |
| 4 | concurrency_ratio (eta) | 1/capacity; measures parallelism utilization |
| 5 | critical_path_latency() | Analyzes DAG for longest dependency chain |
| 6 | async_stage_handler() | Factory: wraps decompose/handler/join into stage handler |
| 7 | capacity parameter | Bounds maximum concurrent forks |

## Data Flow

```
AsyncOrganizer(capacity=4)
       ↓
organizer.fork(
  task="Analyze competitor landscape",
  sub_queries=["pricing data", "feature comparison", "market share"],
  handler=lambda q: f"Analysis of: {q}",
)
       ↓
AsyncThinkResult:
  ├─ outputs: ["Analysis of: pricing data", ...]  (3 items)
  ├─ fork_count: 3
  └─ concurrency_ratio: 0.25  (1/capacity)
       ↓
critical_path_latency(DAG):
  research → analyze → report → review
  └─ CPL = 4.0  (linear chain)
       ↓
async_stage_handler(organizer, decompose, handler, join)
  input: "find papers; summarize findings; write report"
       ↓ decompose (split on ";")
  ["find papers", "summarize findings", "write report"]
       ↓ handler (each .upper())
  ["FIND PAPERS", "SUMMARIZE FINDINGS", "WRITE REPORT"]
       ↓ join (" | ".join)
  output: "FIND PAPERS | SUMMARIZE FINDINGS | WRITE REPORT"
  async_think: {fork_count: 3, concurrency_ratio: 0.25}
```

## Critical Path Analysis

| DAG Node | Dependencies | Depth |
|----------|-------------|-------|
| research | (none) | 1 |
| analyze | research | 2 |
| report | analyze | 3 |
| review | report | 4 |
| **CPL** | | **4.0** |
