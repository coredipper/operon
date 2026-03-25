"""
Example 82 — Managed Organism
===============================

Demonstrates the managed_organism() factory: one function call wires the
full v0.19-0.23 stack (library, watcher, substrate, development, social
learning). Compare this with manually wiring 5-7 components.

Usage:
    python examples/82_managed_organism.py
"""

from operon_ai import (
    BiTemporalMemory,
    MockProvider,
    Nucleus,
    PatternLibrary,
    PatternRunRecord,
    PatternTemplate,
    SkillStage,
    TaskFingerprint,
    Telomere,
    managed_organism,
    consolidate,
)

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

lib = PatternLibrary()
lib.register_template(PatternTemplate(
    template_id="pipeline",
    name="Enterprise Pipeline",
    topology="skill_organism",
    stage_specs=(
        {"name": "intake", "role": "Normalizer"},
        {"name": "process", "role": "Processor"},
    ),
    intervention_policy={},
    fingerprint=TaskFingerprint("sequential", 2, 2, ("normalizer", "processor")),
))
lib.record_run(PatternRunRecord(
    record_id=lib.make_id(), template_id="pipeline",
    fingerprint=TaskFingerprint("sequential", 2, 2, ("normalizer", "processor")),
    success=True, latency_ms=100, tokens_used=500,
))

fast = Nucleus(provider=MockProvider(responses={}))
deep = Nucleus(provider=MockProvider(responses={}))

# ---------------------------------------------------------------------------
# 2. One function call — full stack
# ---------------------------------------------------------------------------

m = managed_organism(
    task="Process quarterly report",
    library=lib,
    fingerprint=TaskFingerprint("sequential", 2, 2, ("normalizer", "processor")),
    fast_nucleus=fast,
    deep_nucleus=deep,
    handlers={
        "intake": lambda task: {"parsed": task},
        "process": lambda task, state, outputs: "Report processed successfully.",
    },
    substrate=BiTemporalMemory(),
    telomere=Telomere(max_operations=100),
    organism_id="org-A",
)

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

result = m.run("Process quarterly report")

print("=== Run Result ===")
print(f"  Output: {result.run_result.final_output}")
print(f"  Template: {result.template_used.name if result.template_used else 'none'}")
print(f"  Watcher: {result.watcher_summary}")
print(f"  Development: {result.development_status.stage.value if result.development_status else 'none'}")
print()

# ---------------------------------------------------------------------------
# 4. Consolidate
# ---------------------------------------------------------------------------

cr = m.consolidate()
print(f"=== Consolidation ===")
print(f"  Result: {cr}")
print()

# ---------------------------------------------------------------------------
# 5. Status
# ---------------------------------------------------------------------------

print("=== Status ===")
for key, value in m.status().items():
    print(f"  {key}: {value}")
print()

# ---------------------------------------------------------------------------
# 6. Export templates (social learning)
# ---------------------------------------------------------------------------

exchange = m.export_templates()
print(f"=== Export ===")
print(f"  Templates: {len(exchange.templates) if exchange else 0}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert result.run_result.final_output is not None
assert result.template_used is not None
assert result.watcher_summary is not None
assert result.development_status is not None
assert cr is not None
assert exchange is not None
print("\n--- all assertions passed ---")
