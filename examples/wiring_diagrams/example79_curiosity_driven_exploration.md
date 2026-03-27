# Example 79: Curiosity-Driven Exploration

## Wiring Diagram

```mermaid
flowchart TB
    subgraph input["Input"]
        task["Customer report"] --> intake
    end

    subgraph pipeline["Skill Organism Pipeline"]
        intake["intake<br/>SkillStage: Normalizer<br/>handler=lambda"] -->|parsed dict| classifier
        classifier["classifier<br/>SkillStage: Router<br/>mode=fast"] -->|category| analyst
        analyst["analyst<br/>SkillStage: Analyst<br/>mode=deep"] --> output["Final Output"]
    end

    subgraph nuclei["Nuclei"]
        fast["Nucleus: fast<br/>MockProvider"] -.-> classifier
        deep["Nucleus: deep<br/>MockProvider"] -.-> analyst
    end

    subgraph curiosity["Curiosity Mechanism"]
        watcher["WatcherComponent<br/>curiosity_threshold=0.5"]
        epi["EpiplexityMonitor<br/>(when attached)"] -->|EXPLORING status| watcher
        watcher -->|"curiosity signal<br/>category=EPISTEMIC"| decision{Signal > threshold<br/>AND model=fast?}
        decision -->|yes| escalate["ESCALATE<br/>→ deep model"]
        decision -->|no| continue["Continue<br/>on current model"]
    end

    watcher -.->|observe| intake
    watcher -.->|observe| classifier
    watcher -.->|observe| analyst

    style escalate fill:#fff9c4
    style output fill:#c8e6c9
    style epi fill:#f3e5f5
    style watcher fill:#e0f7fa
```

```
[Task: "Customer reports unexpected charge on account #1234."]
       |
       v  (U=UNTRUSTED)
  [intake] handler=lambda  →  {"parsed": task}
       |
       v  (V=VALIDATED)
  [classifier] Nucleus=fast  →  "EXECUTE: billing"
       |
       |  ← WatcherComponent observes
       |     └─ If EpiplexityMonitor attached:
       |        status=EXPLORING + curiosity > 0.5 + model=fast
       |        → ESCALATE to deep model
       |
       v  (T=TRUSTED)
  [analyst] Nucleus=deep  →  "EXECUTE: Comprehensive deep analysis."
       |
       v
  [final_output]
```

## Key Patterns

### Curiosity-Triggered Model Escalation
When the EpiplexityMonitor detects high novelty (EXPLORING status), the watcher
emits a curiosity signal. If the signal value exceeds the threshold and the
current stage is running on a fast model, the watcher triggers an ESCALATE
intervention to switch to the deep model for more thorough investigation.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | WatcherComponent | Monitors stages, emits curiosity signals |
| 2 | WatcherConfig | Sets curiosity_escalation_threshold (0.5) |
| 3 | EpiplexityMonitor | Detects novelty/exploration status (when attached) |
| 4 | ESCALATE intervention | Promotes fast-model stage to deep-model execution |
| 5 | SignalCategory.EPISTEMIC | Curiosity signals are classified as epistemic |

### Biological Parallel
Mirrors dopaminergic curiosity in biological systems: novel stimuli trigger
increased attention and resource allocation. A fast "gut reaction" (System A)
is escalated to slow deliberation (System B) when the organism encounters
something genuinely new, similar to how surprising inputs engage the prefrontal
cortex.

## Data Flow

```
str (raw task)
  └─ "Customer reports unexpected charge on account #1234."
       ↓
intake handler (lambda)
  └─ {"parsed": task}
       ↓
classifier (Nucleus: fast)
  └─ "EXECUTE: billing"
       ↓
analyst (Nucleus: deep)
  └─ "EXECUTE: Comprehensive deep analysis."
       ↓
RunResult
  ├─ final_output: str
  ├─ stage_results: list[StageResult] (3 stages)
  └─ (watcher)
       ├─ curiosity signals: 0 (no monitor attached)
       └─ curiosity_escalation_threshold: 0.5
```

## Pipeline Stages

| Stage | Mechanism | Input | Output | Curiosity Behavior |
|-------|-----------|-------|--------|-------------------|
| intake | lambda handler | raw task str | {"parsed": task} | Observed by watcher |
| classifier | Nucleus (fast) | parsed dict | category string | Escalation candidate if curiosity fires |
| analyst | Nucleus (deep) | category | deep analysis | Already on deep model |
