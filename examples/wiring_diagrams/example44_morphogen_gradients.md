# Example 44: Morphogen Gradients

Multi-cellular coordination through shared context variables. Inspired by
embryonic development where cells coordinate via diffusible morphogens.

```mermaid
flowchart TB
    subgraph Orchestrator["Gradient Orchestrator"]
        step[Step Result] --> update[Update Gradients]
        update --> gradient[Morphogen Gradient]
        gradient --> hints[Strategy Hints]
        gradient --> phenotype[Phenotype Config]
        hints --> prompt[Agent Prompt]
        phenotype --> prompt
    end

    subgraph MorphogenTypes["Morphogen Types"]
        direction LR
        complexity[Complexity]
        confidence[Confidence]
        budget[Token Budget]
        error_rate[Error Rate]
        urgency[Urgency]
        risk[Risk Level]
    end

    subgraph Coordination["Multi-Agent Coordination"]
        gradient --> agent1[Agent 1]
        gradient --> agent2[Agent 2]
        gradient --> agent3[Agent 3]
        agent1 --> step
        agent2 --> step
        agent3 --> step
    end

    style gradient fill:#e0f7fa,stroke:#0097a7
```

## Gradient-Based Phenotype Adaptation

```mermaid
flowchart LR
    subgraph HighComplexity["High Complexity Gradient"]
        hc_hint["Use detailed reasoning"]
        hc_temp["temperature: 0.8"]
        hc_tokens["max_tokens: 1500"]
    end

    subgraph LowConfidence["Low Confidence Gradient"]
        lc_hint["Break into sub-steps"]
        lc_temp["temperature: 0.5"]
        lc_tokens["max_tokens: 1000"]
    end

    subgraph LowBudget["Low Budget Gradient"]
        lb_hint["Be concise"]
        lb_temp["temperature: 0.3"]
        lb_tokens["max_tokens: 500"]
    end
```

## ASCII Wiring

```
                              [GradientOrchestrator]
                                       |
              +------------------------+------------------------+
              |                        |                        |
         [complexity]             [confidence]              [budget]
              |                        |                        |
              v                        v                        v
[agent_1] <--hints-- [gradient] --hints--> [agent_2] <--hints-- [agent_3]
    |                     ^                     |                    |
    |                     |                     |                    |
    +-----step_result-----+-----step_result-----+----step_result-----+

Coordination without central control: agents read local gradient concentrations
```

## Morphogen Levels

Each morphogen has three levels based on concentration:

| Morphogen | LOW (< 0.3) | MEDIUM (0.3-0.7) | HIGH (> 0.7) |
|-----------|-------------|------------------|--------------|
| Complexity | Simple task | Moderate | Break down needed |
| Confidence | Uncertain | Stable | Proceed confidently |
| Budget | Abundant | Normal | Conserve tokens |
| Error Rate | Healthy | Concerning | Retry/simplify |
| Urgency | Relaxed | Normal | Expedite |
| Risk | Safe | Caution | Extra validation |

## Key Insight

Agents can coordinate through shared context variables (the "gradient") without
explicit communication between them. Each agent reads its local concentration
and adapts its behavior accordingly—just like cells in an embryo.

## Quorum Trigger

When `should_recruit_help()` returns True (low confidence + high error rate),
it signals that the task should escalate to Quorum Sensing for multi-agent
consensus.

Legend: U = UNTRUSTED, V = VALIDATED, T = TRUSTED.
