# Example 78: Social Learning

## Wiring Diagram

```mermaid
flowchart TB
    subgraph orgA["Organism A (Teacher)"]
        libA["PatternLibrary<br/>template: customer_support<br/>80% success rate"] --> slA["SocialLearning<br/>organism-A"]
        slA --> export["export_templates()<br/>min_success_rate=0.6"]
        export --> exchange["TemplateExchange<br/>peer_id, templates, records"]
    end

    subgraph orgB["Organism B (High Trust)"]
        exchange --> slB["SocialLearning<br/>organism-B"]
        slB --> trust_check{"TrustRegistry<br/>default_trust=0.5"}
        trust_check -->|trust >= 0.3| adopt["ADOPT<br/>customer_support"]
        trust_check -->|trust < 0.3| rejectB["REJECT"]
        adopt --> outcome["record_adoption_outcome()"]
        outcome -->|success| trust_up["trust ++"]
        outcome -->|failure| trust_down["trust --"]
    end

    subgraph orgC["Organism C (Low Trust)"]
        exchange --> slC["SocialLearning<br/>organism-C<br/>default_trust=0.1"]
        slC --> trust_checkC{"TrustRegistry<br/>min_trust=0.3"}
        trust_checkC -->|"0.1 < 0.3"| rejectC["REJECT ALL"]
    end

    style adopt fill:#c8e6c9
    style rejectB fill:#ffcdd2
    style rejectC fill:#ffcdd2
    style exchange fill:#fff9c4
    style trust_up fill:#c8e6c9
    style trust_down fill:#ffcdd2
```

```
Organism A (teacher, 80% success)
  └─ SocialLearning.export_templates(min_success_rate=0.6)
       └─ TemplateExchange {peer_id, templates[], records[]}
                    |
          +---------+---------+
          |                   |
          v                   v
  Organism B               Organism C
  (trust=0.5)             (trust=0.1)
       |                       |
  [TrustRegistry]         [TrustRegistry]
  trust(0.5) >= 0.3       trust(0.1) < 0.3
       |                       |
  ADOPT template          REJECT ALL
       |
  record_adoption_outcome()
  success → trust ↑
  failure → trust ↓
```

## Key Patterns

### Cross-Organism Template Sharing with Epistemic Vigilance
Organisms export proven templates (above a success threshold) and import them
with trust-weighted filtering. The TrustRegistry enforces a minimum trust score
before adoption, and trust updates based on adoption outcomes.

| # | Motif | Role in Pipeline |
|---|-------|-----------------|
| 1 | PatternLibrary | Stores templates with run history and success rates |
| 2 | SocialLearning | Manages export/import of templates between organisms |
| 3 | TrustRegistry | Gates adoption by trust score, adjusts on outcomes |
| 4 | TemplateExchange | Serializable bundle of templates + records for transfer |
| 5 | Provenance tracking | Records which organism a template originated from |

### Biological Parallel
Mirrors cultural transmission in social species: successful behaviors are shared
(teaching), but learners apply epistemic vigilance (trust filtering) before
adopting new behaviors. Repeated successful adoption increases trust; failures
decrease it.

## Data Flow

```
PatternLibrary (Organism A)
  ├─ PatternTemplate("customer_support")
  │   ├─ stages: classify → resolve → verify
  │   └─ tags: ("support", "customer")
  └─ PatternRunRecord[] (4 success, 1 failure)
       ↓
SocialLearning.export_templates(min_success_rate=0.6)
       ↓
TemplateExchange
  ├─ peer_id: "organism-A"
  ├─ templates: [PatternTemplate]
  └─ records: [PatternRunRecord]
       ↓
SocialLearning.import_from_peer(exchange)
       ↓
ImportResult
  ├─ adopted_template_ids: list[str]
  ├─ rejected_template_ids: list[str]
  └─ trust_score_used: float
```

## Pipeline Stages

| Stage | Mechanism | Input | Output | Fallback |
|-------|-----------|-------|--------|----------|
| Export | SocialLearning.export_templates | PatternLibrary + threshold | TemplateExchange | Empty exchange if none qualify |
| Trust Gate | TrustRegistry | Exchange + peer trust | Allow/Reject per template | Reject if below min_trust |
| Adopt | SocialLearning.import_from_peer | Trusted exchange | ImportResult | Reject untrusted |
| Outcome | record_adoption_outcome | Success/failure signal | Updated trust score | Trust decays on failure |
