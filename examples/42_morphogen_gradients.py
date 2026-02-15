"""
Example 42: Morphogen Gradients - Multi-Cellular Coordination
==============================================================

Demonstrates the MorphogenGradient module for coordinating multi-agent systems:

1. **Morphogen Types**: Complexity, Confidence, Budget, Error Rate, Urgency, Risk
2. **Gradient Orchestrator**: Updates gradients based on agent performance
3. **Phenotype Adaptation**: Agents adjust behavior based on local concentrations
4. **Strategy Hints**: Automatic hints for agent prompts

Biological Analogy:
In embryonic development, cells coordinate through morphogen gradients -
diffusible signaling molecules whose concentration varies spatially. Cells
read their local concentration and differentiate accordingly, enabling
pattern formation without a central controller.

The key insight: Agents can coordinate through shared context variables
(the "gradient") without explicit communication between them.

References:
- Article Section 6.5.2: Morphogen Gradients - Coordination Without Central Control
"""

from operon_ai.coordination import (
    MorphogenType,
    MorphogenGradient,
    GradientOrchestrator,
    PhenotypeConfig,
)


def main():
    try:
        print("=" * 60)
        print("Morphogen Gradients - Multi-Cellular Coordination")
        print("=" * 60)

        # =================================================================
        # SECTION 1: Basic Gradient Setup
        # =================================================================
        print("\n--- 1. BASIC GRADIENT ---")
        print("Setting up morphogen concentrations...\n")

        gradient = MorphogenGradient()

        # Show default values
        print("  Default gradient concentrations:")
        for mtype in MorphogenType:
            value = gradient.get(mtype)
            level = gradient.get_level(mtype)
            print(f"    {mtype.value:12} = {value:.2f} ({level})")

        # Modify for a complex, low-confidence scenario
        gradient.set(MorphogenType.COMPLEXITY, 0.8)
        gradient.set(MorphogenType.CONFIDENCE, 0.2)
        gradient.set(MorphogenType.ERROR_RATE, 0.3)

        print("\n  After adjustment (complex task, low confidence):")
        for mtype in MorphogenType:
            value = gradient.get(mtype)
            level = gradient.get_level(mtype)
            print(f"    {mtype.value:12} = {value:.2f} ({level})")

        # =================================================================
        # SECTION 2: Strategy Hints
        # =================================================================
        print("\n--- 2. STRATEGY HINTS ---")
        print("Generating hints based on gradient...\n")

        hints = gradient.get_strategy_hints()
        if hints:
            print("  Generated hints:")
            for hint in hints:
                print(f"    - {hint}")
        else:
            print("  No special hints needed (neutral gradient)")

        # =================================================================
        # SECTION 3: Context Injection
        # =================================================================
        print("\n--- 3. CONTEXT INJECTION ---")
        print("Generating agent context preamble...\n")

        context = gradient.get_context_injection()
        print("  Context for agent prompt:")
        for line in context.split('\n')[:5]:
            print(f"    {line}")
        print("    ...")

        # =================================================================
        # SECTION 4: Gradient Orchestrator
        # =================================================================
        print("\n--- 4. ORCHESTRATOR ---")
        print("Simulating agent steps with automatic gradient updates...\n")

        orchestrator = GradientOrchestrator(silent=True)

        # Simulate a sequence of agent steps
        steps = [
            {"success": True, "tokens_used": 200, "total_budget": 1000},
            {"success": True, "tokens_used": 300, "total_budget": 1000},
            {"success": False, "tokens_used": 150, "total_budget": 1000},  # Failure
            {"success": True, "tokens_used": 200, "total_budget": 1000},
        ]

        print("  Simulating agent steps:")
        for i, step in enumerate(steps):
            orchestrator.report_step_result(**step)
            grad = orchestrator.gradient

            status = "success" if step["success"] else "FAILURE"
            print(f"    Step {i+1} ({status:7}): "
                  f"confidence={grad.get(MorphogenType.CONFIDENCE):.2f} "
                  f"error_rate={grad.get(MorphogenType.ERROR_RATE):.2f} "
                  f"budget={grad.get(MorphogenType.BUDGET):.2f}")

        # =================================================================
        # SECTION 5: Phenotype Parameters
        # =================================================================
        print("\n--- 5. PHENOTYPE ADAPTATION ---")
        print("Calculating agent parameters from gradient...\n")

        params = orchestrator.get_phenotype_params()
        print("  Phenotype parameters:")
        print(f"    Temperature: {params['temperature']:.2f}")
        print(f"    Max tokens: {params['max_tokens']}")
        print(f"    Verification threshold: {params['verification_threshold']:.2f}")

        # Custom phenotype config
        print("\n  With custom phenotype config:")
        custom_config = PhenotypeConfig(
            low_complexity_temperature=0.1,
            high_complexity_temperature=1.0,
            critical_budget_max_tokens=50,
        )
        orchestrator.phenotype_config = custom_config
        params = orchestrator.get_phenotype_params()
        print(f"    Temperature: {params['temperature']:.2f}")
        print(f"    Max tokens: {params['max_tokens']}")

        # =================================================================
        # SECTION 6: Coordination Decisions
        # =================================================================
        print("\n--- 6. COORDINATION DECISIONS ---")
        print("Gradient-based coordination signals...\n")

        # Set up a stressed scenario
        orchestrator.gradient.set(MorphogenType.CONFIDENCE, 0.2)
        orchestrator.gradient.set(MorphogenType.ERROR_RATE, 0.6)
        orchestrator.gradient.set(MorphogenType.COMPLEXITY, 0.8)
        orchestrator.gradient.set(MorphogenType.BUDGET, 0.15)

        print("  Stressed scenario gradient:")
        print(f"    Confidence: {orchestrator.gradient.get(MorphogenType.CONFIDENCE):.2f}")
        print(f"    Error rate: {orchestrator.gradient.get(MorphogenType.ERROR_RATE):.2f}")
        print(f"    Complexity: {orchestrator.gradient.get(MorphogenType.COMPLEXITY):.2f}")
        print(f"    Budget: {orchestrator.gradient.get(MorphogenType.BUDGET):.2f}")

        print(f"\n  Coordination signals:")
        print(f"    Should recruit help (Quorum): {orchestrator.should_recruit_help()}")
        print(f"    Should reduce capabilities: {orchestrator.should_reduce_capabilities()}")

        # =================================================================
        # SECTION 7: Full Agent Context
        # =================================================================
        print("\n--- 7. FULL AGENT CONTEXT ---")
        print("Complete context injection for agent...\n")

        full_context = orchestrator.get_agent_context()
        print("  Agent context (truncated):")
        lines = full_context.split('\n')
        for line in lines[:10]:
            print(f"    {line}")
        if len(lines) > 10:
            print("    ...")

        # =================================================================
        # SECTION 8: Statistics
        # =================================================================
        print("\n--- 8. STATISTICS ---")
        stats = orchestrator.stats()

        print(f"  Total gradient updates: {stats['total_updates']}")
        print(f"  Should recruit help: {stats['should_recruit_help']}")
        print(f"  Should reduce capabilities: {stats['should_reduce_capabilities']}")
        print("\n  Current gradient:")
        for k, v in stats['current_gradient'].items():
            print(f"    {k}: {v:.2f}")

        print("\n" + "=" * 60)
        print("Morphogen gradients enable coordination without central control!")
        print("=" * 60)

    except Exception as e:
        print(f"\n Error during example: {e}")
        raise


if __name__ == "__main__":
    main()
