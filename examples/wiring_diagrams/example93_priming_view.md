# Example 93: PrimingView

## Wiring Diagram

```
SubstrateView                        PrimingView
  (facts, query,                     (5 channels)
   record_time)                           ^
        |                                 |
        v                                 |
  [build_priming_view]  ─────────────────┘
        ^
        |
  additional channels:
  ├── recent_outputs   (stage results)
  ├── telemetry        (latency, events)
  ├── experience       (past actions + outcomes)
  ├── trust_context    (peer trust scores)
  └── developmental_status (stage gating)
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | SubstrateView | Base view: facts + query + record_time |
| 2 | build_priming_view() | Promotes SubstrateView to PrimingView with channels |
| 3 | PrimingView | Extended view with 5 priming channels |
| 4 | facts channel | Bi-temporal facts from substrate |
| 5 | recent_outputs channel | Previous stage outputs for context |
| 6 | telemetry channel | Performance events (latency, completion) |
| 7 | experience channel | Past action-outcome pairs |
| 8 | trust_context channel | Peer trust scores for social learning |
| 9 | Auto-freeze | __post_init__ converts dicts to MappingProxyType |

## Data Flow

```
SubstrateView(facts=(), query=None, record_time=now)
       ↓
build_priming_view(
  base,
  recent_outputs=({"stage": "research", "output": "Found 3 papers"},),
  trust_context={"peer_A": 0.85, "peer_B": 0.42},
  developmental_status=None,
)
       ↓
PrimingView (isinstance SubstrateView = True)
  ├─ facts:                ()          ← from base
  ├─ recent_outputs:       1 entry     ← stage results
  ├─ telemetry:            ()          ← (default empty)
  ├─ experience:           ()          ← (default empty)
  ├─ trust_context:        {peer_A: 0.85, peer_B: 0.42}
  └─ developmental_status: None
```

## Channel Summary

| Channel | Type | Source | Purpose |
|---------|------|--------|---------|
| facts | tuple[dict] | BiTemporalMemory | Known facts at query time |
| recent_outputs | tuple[dict] | Previous stages | Context chaining |
| telemetry | tuple[dict] | Runtime | Performance monitoring |
| experience | tuple[dict] | Past runs | Action-outcome learning |
| trust_context | dict[str, float] | Social learning | Peer reliability scores |
