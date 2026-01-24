# Example 38: Linear Budget Tracking

Tracks Claude Code token costs against Linear ticket estimates using git commit
correlation and ATP_Store for metabolic budget management.

```mermaid
flowchart TB
    subgraph inputs["Data Sources"]
        CSV[("Anthropic CSV<br/>cost/user/day")]
        GIT[("Git History<br/>commits/branches")]
        LINEAR[("Linear API<br/>tickets/estimates")]
    end

    subgraph parsing["Parsing Layer"]
        CSV --> CSVParser["CostCSVParser<br/>━━━━━━━━━━━━━<br/>• Extract username<br/>• Aggregate cost"]
        GIT --> GitAnalyzer["GitTicketAnalyzer<br/>━━━━━━━━━━━━━<br/>• Parse branch → ticket<br/>• Map commits to dates"]
    end

    subgraph attribution["Attribution"]
        CSVParser --> Attributor["CostAttributor<br/>━━━━━━━━━━━━━<br/>• Correlate user+date<br/>• Split by commit count"]
        GitAnalyzer --> Attributor
    end

    subgraph budgeting["Budget Tracking"]
        LINEAR --> BudgetConfig["BudgetConfig<br/>$/point multipliers"]
        BudgetConfig --> Tracker["TicketBudgetTracker<br/>ATP_Store per ticket"]
        Attributor --> Tracker
    end

    subgraph states["Metabolic States"]
        Tracker --> Normal["NORMAL<br/>< 70%"]
        Tracker --> Conserving["CONSERVING<br/>70-90%"]
        Tracker --> Starving["STARVING<br/>> 90%"]
        Tracker --> Over["OVER BUDGET<br/>> 100%"]
    end

    style CSV fill:#e1f5fe
    style GIT fill:#e1f5fe
    style LINEAR fill:#e1f5fe
    style Normal fill:#c8e6c9
    style Conserving fill:#fff9c4
    style Starving fill:#ffccbc
    style Over fill:#ffcdd2
```

## ASCII Wiring

```
[csv] --cost(U)--> [csv_parser] --cost(V)--+
                                           +--> [attributor] --ticket_cost(V)--> [tracker] --> [metabolic_state]
[git] --commits(U)--> [git_analyzer] --commits(V)--+                              ^
                                                                                   |
[linear] --estimates(T)------------------------------------------------------------+
```

## Key Concepts

- **Cost Attribution**: Maps API costs to Linear tickets via git commit correlation
- **Metabolic States**: NORMAL → CONSERVING → STARVING → OVER_BUDGET
- **Ischemia Detection**: Predictive alerts before budget exhaustion

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
