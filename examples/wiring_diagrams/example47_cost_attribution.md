# Example 47: Enhanced Cost Attribution

## Wiring Diagram

```mermaid
flowchart TB
    subgraph inputs["Data Sources"]
        CSV[("Anthropic CSV<br/>cost/user/day")]
        GIT[("Git History<br/>commits/branches")]
        LINEAR[("Linear API<br/>tickets/teams")]
    end

    subgraph parsing["Stage 1: Parse"]
        CSV --> CSVParser["CostCSVParser<br/>━━━━━━━━━━━━━<br/>• Extract user<br/>• Aggregate by day"]
        GIT --> GitAnalyzer["GitTicketAnalyzer<br/>━━━━━━━━━━━━━<br/>• Branch → ticket<br/>• Commit mapping"]
        LINEAR --> LinearClient["LinearClient<br/>━━━━━━━━━━━━━<br/>• Ticket metadata<br/>• Team assignment"]
    end

    subgraph attribution["Stage 2: Attribute"]
        CSVParser --> Attributor
        GitAnalyzer --> Attributor
        LinearClient --> Attributor
        Attributor["CostAttributor<br/>━━━━━━━━━━━━━<br/>• Correlate user+date<br/>• Split by commits"]
    end

    subgraph tracking["Stage 3: Track"]
        Attributor --> TeamTracker["TeamBudgetTracker<br/>━━━━━━━━━━━━━<br/>• ATP_Store per team<br/>• State transitions"]
        TeamTracker --> Gradient["MorphogenGradient<br/>━━━━━━━━━━━━━<br/>• Budget signals<br/>• Cross-team coord"]
    end

    subgraph analysis["Stage 4: Analyze"]
        TeamTracker --> Trend["TrendAnalyzer<br/>━━━━━━━━━━━━━<br/>• Week-over-week<br/>• Burn rate<br/>• Exhaustion predict"]
    end

    subgraph output["Output"]
        Trend --> Report[("CostReport<br/>teams, tickets<br/>trends, alerts")]
        Report --> Markdown["MarkdownReport<br/>━━━━━━━━━━━━━<br/>• Summary tables<br/>• Alert sections"]
    end

    style CSV fill:#e1f5fe
    style GIT fill:#e1f5fe
    style LINEAR fill:#e1f5fe
    style Report fill:#c8e6c9
```

## Key Patterns

### Morphogen Gradients
Each team's budget status is broadcast as a morphogen signal.
Other teams can sense "metabolic scarcity" and adjust behavior.
This enables coordination without central control.

### ATP Budget per Team
Each team has an ATP_Store representing their monthly budget.
State transitions (NORMAL → CONSERVING → STARVING) trigger alerts.

### Trend Analysis
- Week-over-week comparison detects acceleration
- Daily burn rate predicts exhaustion date
- Alerts trigger when projection < 14 days

## Data Flow

```
TicketCost
  ├─ ticket_id: str
  ├─ team_id: str
  ├─ estimate_points: int
  ├─ budget_usd: float
  ├─ spent_usd: float
  └─ state: MetabolicState
       ↓
TeamSummary
  ├─ team_id: str
  ├─ budget_usd: float
  ├─ spent_usd: float
  ├─ utilization: float
  └─ over_budget_tickets: int
       ↓
CostReport
  ├─ org_total_budget: float
  ├─ org_total_spent: float
  ├─ teams: list[TeamSummary]
  ├─ tickets: list[TicketCost]
  ├─ trend: TrendData
  └─ alerts: list[tuple[AlertLevel, str]]
```

## Budget States

| State | Utilization | Behavior |
|-------|-------------|----------|
| NORMAL | < 70% | Full operation |
| CONSERVING | 70-90% | Warning alerts |
| STARVING | > 90% | Critical alerts |
| FEASTING | Recent large decrease | (transient) |
