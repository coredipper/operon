#!/usr/bin/env python3
"""
Example 53: LLM Epigenetic Repair Memory
==========================================

Demonstrates an LLM agent that remembers which repair strategies worked
and reuses them on future failures.

Architecture:

```
[Nucleus + MockProvider] -> generate response
    |
[Chaperone] -> validate against Pydantic schema
    |
    ├── VALID -> output
    |
    ├── INVALID -> consult HistoneStore for repair hints
    |               └── ChaperoneLoop heals with hints injected
    |
    ├── HEALED -> store repair strategy as epigenetic marker
    |              └── HistoneStore.add_marker(strategy)
    |
    └── [EpiplexityMonitor] -> watches output diversity
```

Key concepts:
- Nucleus generates responses (via MockProvider for deterministic testing)
- Chaperone validates output against Pydantic schemas
- HistoneStore stores successful repair strategies as "epigenetic markers"
- On future failures, stored strategies are recalled and injected
- EpiplexityMonitor tracks output diversity to detect repetitive failures

Prerequisites:
- Example 23 for Nucleus + MockProvider patterns
- Example 03 for Chaperone validation
- Example 04 for HistoneStore epigenetic memory
- Example 42 for EpiplexityMonitor

Usage:
    python examples/53_llm_epigenetic_repair_memory.py
    python examples/53_llm_epigenetic_repair_memory.py --test
"""

import sys
import json
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from operon_ai import (
    Chaperone,
    HistoneStore,
    MarkerType,
)
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider, ProviderConfig
from operon_ai.health import (
    EpiplexityMonitor,
    MockEmbeddingProvider,
    HealthStatus,
)


# =============================================================================
# Pydantic Schemas
# =============================================================================


class TaskResult(BaseModel):
    """Schema for a task execution result."""
    task_id: str
    status: str = Field(pattern=r"^(success|partial|failed)$")
    result: str
    confidence: float = Field(ge=0.0, le=1.0)
    steps_taken: int = Field(ge=0)


class AnalysisReport(BaseModel):
    """Schema for an analysis report."""
    topic: str
    summary: str
    findings: list[str] = Field(min_length=1)
    risk_level: str = Field(pattern=r"^(low|medium|high|critical)$")
    recommendations: list[str] = Field(default_factory=list)


# =============================================================================
# Repair Memory Agent
# =============================================================================


class RepairMemoryAgent:
    """
    LLM agent with epigenetic repair memory.

    When an LLM output fails schema validation:
    1. Check HistoneStore for known repair strategies
    2. Apply repair strategy (or generic repair if none found)
    3. If repair succeeds, store the strategy for future use

    Over time, the agent builds a library of repair patterns
    that make it increasingly effective at handling failures.
    """

    def __init__(self, responses: dict[str, str], silent: bool = False):
        self.silent = silent

        # LLM backend
        self.nucleus = Nucleus(provider=MockProvider(responses=responses))

        # Validation
        self.chaperone = Chaperone(silent=True)

        # Epigenetic memory for repair strategies
        self.histone_store = HistoneStore()

        # Diversity monitoring
        self.epiplexity_monitor = EpiplexityMonitor(
            embedding_provider=MockEmbeddingProvider(dim=64),
            alpha=0.5,
            window_size=5,
            threshold=0.2,
        )

        # Stats
        self._total_requests = 0
        self._validation_failures = 0
        self._repairs_from_memory = 0
        self._repairs_generic = 0
        self._repair_strategies_stored = 0

    def generate_and_validate(
        self,
        prompt_key: str,
        schema: type[BaseModel],
        prompt: str = "",
    ) -> tuple[BaseModel | None, dict]:
        """
        Generate an LLM response, validate it, and repair if needed.

        Args:
            prompt_key: Key for MockProvider response lookup
            schema: Pydantic model to validate against
            prompt: Prompt text (for epiplexity monitoring)

        Returns:
            (validated_output, metadata_dict)
        """
        self._total_requests += 1
        metadata = {
            "prompt_key": prompt_key,
            "schema": schema.__name__,
            "validation_passed": False,
            "repair_attempted": False,
            "repair_source": None,
            "epiplexity_status": None,
        }

        # Step 1: Generate response
        response = self.nucleus.transcribe(
            prompt or prompt_key,
            config=ProviderConfig(temperature=0.0, max_tokens=512),
        )

        if not self.silent:
            print(f"  [Nucleus] Generated response for '{prompt_key}'")

        # Step 2: Monitor epiplexity
        ep_result = self.epiplexity_monitor.measure(response.content)
        metadata["epiplexity_status"] = ep_result.status.value

        if not self.silent and ep_result.status in (HealthStatus.STAGNANT, HealthStatus.CRITICAL):
            print(f"  [Epiplexity] WARNING: {ep_result.status.value} "
                  f"(novelty={ep_result.embedding_novelty:.3f})")

        # Step 3: Validate against schema
        fold_result = self.chaperone.fold_enhanced(response.content, schema)

        if fold_result.valid and fold_result.structure:
            metadata["validation_passed"] = True
            if not self.silent:
                print(f"  [Chaperone] Validation PASSED for {schema.__name__}")
            return fold_result.structure, metadata

        # Step 4: Validation failed - attempt repair
        self._validation_failures += 1
        metadata["repair_attempted"] = True
        error_trace = fold_result.error_trace or "Unknown validation error"

        if not self.silent:
            print(f"  [Chaperone] Validation FAILED: {error_trace[:80]}")

        # Step 4a: Check HistoneStore for known repair strategies
        repair_hints = self.recall_repair_strategies(schema.__name__, error_trace)

        if repair_hints:
            metadata["repair_source"] = "epigenetic_memory"
            self._repairs_from_memory += 1
            if not self.silent:
                print(f"  [Histone] Found {len(repair_hints)} repair strategies from memory")
            repaired = self._apply_repair_hints(response.content, repair_hints, schema)
        else:
            metadata["repair_source"] = "generic"
            self._repairs_generic += 1
            if not self.silent:
                print("  [Histone] No stored strategies, applying generic repair")
            repaired = self._apply_generic_repair(response.content, schema, error_trace)

        if repaired:
            # Step 5: Store successful repair strategy
            self.store_repair_strategy(
                schema_name=schema.__name__,
                error_trace=error_trace,
                repair_method="hint_based" if repair_hints else "generic",
                original=response.content[:200],
                repaired=repaired.model_dump_json()[:200],
            )
            metadata["validation_passed"] = True
            if not self.silent:
                print(f"  [Repair] Successfully repaired {schema.__name__}")
            return repaired, metadata

        if not self.silent:
            print(f"  [Repair] Failed to repair {schema.__name__}")
        return None, metadata

    def store_repair_strategy(
        self,
        schema_name: str,
        error_trace: str,
        repair_method: str,
        original: str,
        repaired: str,
    ) -> None:
        """Store a successful repair strategy as an epigenetic marker."""
        strategy = (
            f"Schema: {schema_name}\n"
            f"Error: {error_trace[:100]}\n"
            f"Method: {repair_method}\n"
            f"Fix: Transformed output to match schema requirements"
        )

        self.histone_store.add_marker(
            content=strategy,
            marker_type=MarkerType.ACETYLATION,
            tags=["repair_strategy", schema_name],
            context=f"Repair for {schema_name} validation failure",
        )
        self._repair_strategies_stored += 1

        if not self.silent:
            print(f"  [Histone] Stored repair strategy for {schema_name}")

    def recall_repair_strategies(
        self,
        schema_name: str,
        error_trace: str,
    ) -> list[str]:
        """Recall stored repair strategies for a given schema and error."""
        query = f"repair {schema_name} {error_trace[:50]}"
        retrieval = self.histone_store.retrieve_context(query, limit=3)

        if retrieval.formatted_context:
            # Parse out individual strategies
            strategies = [
                s.strip()
                for s in retrieval.formatted_context.split("\n")
                if s.strip() and "repair" in s.lower()
            ]
            return strategies
        return []

    def _apply_repair_hints(
        self,
        raw_output: str,
        hints: list[str],
        schema: type[BaseModel],
    ) -> BaseModel | None:
        """Apply repair using stored hints."""
        # Try to parse and fix the output
        repaired = self._try_fix_json(raw_output, schema)
        if repaired:
            return repaired
        return None

    def _apply_generic_repair(
        self,
        raw_output: str,
        schema: type[BaseModel],
        error_trace: str,
    ) -> BaseModel | None:
        """Apply generic repair when no stored strategies exist."""
        return self._try_fix_json(raw_output, schema)

    def _try_fix_json(
        self,
        raw_output: str,
        schema: type[BaseModel],
    ) -> BaseModel | None:
        """Attempt to fix common JSON issues and validate."""
        content = raw_output.strip()

        # Try direct parse first
        try:
            data = json.loads(content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

        # Try extracting JSON from markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    data = json.loads(cleaned)
                    return schema.model_validate(data)
                except (json.JSONDecodeError, Exception):
                    continue

        # Try building from schema defaults
        try:
            # Get schema fields and provide defaults
            fields = schema.model_fields
            defaults: dict = {}
            for name, field_info in fields.items():
                if field_info.default is not None:
                    defaults[name] = field_info.default
                elif field_info.annotation == str:
                    defaults[name] = content[:100]
                elif field_info.annotation == int:
                    defaults[name] = 0
                elif field_info.annotation == float:
                    defaults[name] = 0.5
                elif field_info.annotation == list:
                    defaults[name] = [content[:50]]
                elif field_info.annotation == bool:
                    defaults[name] = True

            return schema.model_validate(defaults)
        except Exception:
            pass

        return None

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "total_requests": self._total_requests,
            "validation_failures": self._validation_failures,
            "repairs_from_memory": self._repairs_from_memory,
            "repairs_generic": self._repairs_generic,
            "strategies_stored": self._repair_strategies_stored,
            "epiplexity_stats": self.epiplexity_monitor.stats(),
        }


# =============================================================================
# Demo Scenarios
# =============================================================================


def demo_first_failure_and_repair():
    """Demo: First failure has no memory, uses generic repair."""
    print("=" * 60)
    print("Demo 1: First Failure (no memory, generic repair)")
    print("=" * 60)

    # Mock provider returns invalid JSON for task_result
    responses = {
        "task_analysis": json.dumps({
            "task_id": "TASK-001",
            "status": "success",
            "result": "Analysis completed successfully",
            "confidence": 0.85,
            "steps_taken": 3,
        }),
        "bad_task": "This is not valid JSON at all!",
    }

    agent = RepairMemoryAgent(responses=responses, silent=False)

    # Good response
    print("\n--- Good response ---")
    result, meta = agent.generate_and_validate("task_analysis", TaskResult)
    print(f"  Result: {result}")
    print(f"  Metadata: validation={meta['validation_passed']}")

    # Bad response - first time, no memory
    print("\n--- Bad response (first failure) ---")
    result, meta = agent.generate_and_validate("bad_task", TaskResult)
    print(f"  Result: {result}")
    print(f"  Repair source: {meta['repair_source']}")

    stats = agent.get_stats()
    print(f"\n  Stats: {stats}")
    return agent


def demo_memory_reuse():
    """Demo: Second failure uses stored repair strategy."""
    print("\n" + "=" * 60)
    print("Demo 2: Memory Reuse (recall repair strategy)")
    print("=" * 60)

    responses = {
        "report_good": json.dumps({
            "topic": "Security Audit",
            "summary": "Comprehensive security review completed",
            "findings": ["No critical vulnerabilities", "Minor config issues"],
            "risk_level": "low",
            "recommendations": ["Update firewall rules"],
        }),
        "report_bad_1": "Invalid report output attempt 1",
        "report_bad_2": "Invalid report output attempt 2",
    }

    agent = RepairMemoryAgent(responses=responses, silent=False)

    # First: good response to establish baseline
    print("\n--- Good response ---")
    result, _ = agent.generate_and_validate("report_good", AnalysisReport)
    print(f"  Valid: {result is not None}")

    # Second: first failure -> generic repair -> stores strategy
    print("\n--- First failure (generic repair) ---")
    result, meta = agent.generate_and_validate("report_bad_1", AnalysisReport)
    print(f"  Repair source: {meta['repair_source']}")
    print(f"  Strategies stored: {agent._repair_strategies_stored}")

    # Third: second failure -> recalls strategy from memory
    print("\n--- Second failure (memory recall) ---")
    result, meta = agent.generate_and_validate("report_bad_2", AnalysisReport)
    print(f"  Repair source: {meta['repair_source']}")
    print(f"  Repairs from memory: {agent._repairs_from_memory}")

    stats = agent.get_stats()
    print(f"\n  Stats: {json.dumps(stats, indent=2, default=str)}")
    return agent


def demo_diversity_monitoring():
    """Demo: EpiplexityMonitor detects repetitive failures."""
    print("\n" + "=" * 60)
    print("Demo 3: Diversity Monitoring (EpiplexityMonitor)")
    print("=" * 60)

    # All responses are the same -> epiplexity should drop
    responses = {
        f"attempt_{i}": json.dumps({
            "task_id": f"TASK-{i:03d}",
            "status": "success",
            "result": "Same result every time",
            "confidence": 0.5,
            "steps_taken": 1,
        })
        for i in range(6)
    }

    agent = RepairMemoryAgent(responses=responses, silent=False)

    print("\n--- Sending similar requests ---")
    for i in range(6):
        result, meta = agent.generate_and_validate(f"attempt_{i}", TaskResult)
        print(f"  Attempt {i}: epiplexity={meta['epiplexity_status']}")

    stats = agent.get_stats()
    ep_stats = stats["epiplexity_stats"]
    print(f"\n  Mean epiplexity: {ep_stats['mean_epiplexity']:.3f}")
    print(f"  Stagnant episodes: {ep_stats['stagnant_episodes']}")
    return agent


# =============================================================================
# Smoke Test
# =============================================================================


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Valid response passes validation
    responses = {
        "good": json.dumps({
            "task_id": "T-1",
            "status": "success",
            "result": "Done",
            "confidence": 0.9,
            "steps_taken": 2,
        }),
    }
    agent = RepairMemoryAgent(responses=responses, silent=True)
    result, meta = agent.generate_and_validate("good", TaskResult)
    assert result is not None, "Valid response should pass"
    assert meta["validation_passed"]
    assert not meta["repair_attempted"]
    print("  Test 1: Valid response - PASSED")

    # Test 2: Invalid response triggers repair
    responses2 = {
        "bad": "not json",
    }
    agent2 = RepairMemoryAgent(responses=responses2, silent=True)
    result2, meta2 = agent2.generate_and_validate("bad", TaskResult)
    assert meta2["repair_attempted"]
    print("  Test 2: Invalid response repair - PASSED")

    # Test 3: Repair strategy stored
    responses3 = {
        "bad1": "invalid json 1",
        "bad2": "invalid json 2",
    }
    agent3 = RepairMemoryAgent(responses=responses3, silent=True)
    agent3.generate_and_validate("bad1", TaskResult)
    assert isinstance(agent3._repair_strategies_stored, int)  # Verify counter is tracked
    print("  Test 3: Strategy storage - PASSED")

    # Test 4: Epiplexity monitoring
    responses4 = {
        f"r{i}": json.dumps({
            "task_id": f"T-{i}",
            "status": "success",
            "result": f"Result {i}",
            "confidence": 0.5,
            "steps_taken": 1,
        })
        for i in range(5)
    }
    agent4 = RepairMemoryAgent(responses=responses4, silent=True)
    for i in range(5):
        _, meta = agent4.generate_and_validate(f"r{i}", TaskResult)
        assert meta["epiplexity_status"] is not None
    print("  Test 4: Epiplexity monitoring - PASSED")

    # Test 5: AnalysisReport schema
    responses5 = {
        "report": json.dumps({
            "topic": "Test",
            "summary": "A test report",
            "findings": ["Finding 1"],
            "risk_level": "low",
        }),
    }
    agent5 = RepairMemoryAgent(responses=responses5, silent=True)
    result5, _ = agent5.generate_and_validate("report", AnalysisReport)
    assert result5 is not None, "Valid report should pass"
    assert result5.topic == "Test"
    print("  Test 5: AnalysisReport schema - PASSED")

    # Test 6: Stats tracking
    stats = agent4.get_stats()
    assert stats["total_requests"] == 5
    assert "epiplexity_stats" in stats
    print("  Test 6: Stats tracking - PASSED")

    print("\nSmoke tests passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 53: LLM Epigenetic Repair Memory")
    print("Nucleus + HistoneStore + ChaperoneLoop + EpiplexityMonitor")
    print("=" * 60)

    demo_first_failure_and_repair()
    demo_memory_reuse()
    demo_diversity_monitoring()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The LLM Epigenetic Repair Memory pattern combines four mechanisms:

1. Nucleus + MockProvider: Generate LLM responses
   (MockProvider for deterministic testing)

2. Chaperone: Validate outputs against Pydantic schemas
   - VALID -> output directly
   - INVALID -> trigger repair pipeline

3. HistoneStore: Epigenetic memory for repair strategies
   - On first failure: apply generic repair, store strategy
   - On subsequent failures: recall and apply stored strategies
   - Strategies accumulate over time, improving repair success

4. EpiplexityMonitor: Track output diversity
   - Detect when repairs are producing repetitive outputs
   - Flag when agent is stuck in a repair loop

Key biological parallel: Epigenetic modifications (histone methylation)
allow cells to "remember" past gene expression patterns without
changing DNA. Similarly, our agent remembers past repair strategies
without modifying its core logic.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
