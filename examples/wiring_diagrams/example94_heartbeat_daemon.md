# Example 94: HeartbeatDaemon

## Wiring Diagram

```
runs accumulate
  on_run_start() x N
        |
        v
  [HeartbeatDaemon]
    (extends WatcherComponent)
        |
        v
  [heartbeat()] ──── conditions met? ────┐
        |                                 |
        │ NO                         YES  │
        v                                 v
  {triggered: false,             [SleepConsolidation]
   reason: "..."}                  .consolidate()
                                         |
                                         v
                                  ConsolidationResult
                                    templates_created
                                    memories_promoted
                                         |
                                         v
                                  reset run counter
```

## Key Patterns

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | HeartbeatDaemon | WatcherComponent subclass with consolidation trigger |
| 2 | on_run_start() | Increments internal run counter per observed run |
| 3 | heartbeat() | Checks min_runs + cooldown, fires consolidation if met |
| 4 | min_runs_before_consolidate | Threshold: N runs must accumulate before trigger |
| 5 | heartbeat_interval_s | Cooldown: minimum seconds between consolidations |
| 6 | SleepConsolidation | Injected consolidation engine (mock or real) |
| 7 | summary() | Includes heartbeat stats alongside watcher stats |

## Data Flow

```
HeartbeatDaemon(
  consolidation=SleepConsolidation,
  min_runs_before_consolidate=3,
  heartbeat_interval_s=0,
)
       ↓
on_run_start("task_0", {})  → run_count = 1
on_run_start("task_1", {})  → run_count = 2
       ↓
heartbeat()
  └─ run_count (2) < min_runs (3) → {triggered: false}
       ↓
on_run_start("task_2", {})  → run_count = 3
       ↓
heartbeat()
  └─ run_count (3) >= min_runs (3) → trigger!
       ↓
consolidation.consolidate()
  ├─ templates_created: 1
  └─ memories_promoted: 2
       ↓
reset run counter
       ↓
daemon.summary()
  └─ {heartbeat: {total_heartbeats_triggered: 1, ...}}
```

## Trigger Conditions

| Condition | Check | Must Pass |
|-----------|-------|-----------|
| Run count | run_count >= min_runs_before_consolidate | Yes |
| Cooldown | elapsed >= heartbeat_interval_s | Yes |
| Both must hold | AND logic | Yes |
