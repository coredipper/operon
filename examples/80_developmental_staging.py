"""
Example 80 — Developmental Staging
====================================

Demonstrates DevelopmentController: organisms progress through maturity
stages (EMBRYONIC → JUVENILE → ADOLESCENT → MATURE), capabilities are
gated by developmental stage, and critical periods close permanently.

Usage:
    python examples/80_developmental_staging.py
"""

from operon_ai import (
    DevelopmentController,
    DevelopmentConfig,
    DevelopmentalStage,
    CriticalPeriod,
    Telomere,
)

# ---------------------------------------------------------------------------
# 1. Create a telomere with 100 max operations
# ---------------------------------------------------------------------------

t = Telomere(max_operations=100)
t.start()

dc = DevelopmentController(
    telomere=t,
    config=DevelopmentConfig(
        juvenile_threshold=0.10,
        adolescent_threshold=0.35,
        mature_threshold=0.70,
    ),
    critical_periods=(
        CriticalPeriod("rapid_learning", DevelopmentalStage.EMBRYONIC, DevelopmentalStage.JUVENILE, "fast template adoption"),
        CriticalPeriod("tool_exploration", DevelopmentalStage.JUVENILE, DevelopmentalStage.ADOLESCENT, "try new tools"),
    ),
)

# ---------------------------------------------------------------------------
# 2. Tick through lifecycle, show transitions
# ---------------------------------------------------------------------------

print("=== Developmental Lifecycle ===")
last_stage = dc.stage
for i in range(90):
    dc.tick()
    if dc.stage != last_stage:
        print(f"  Tick {i+1}: {last_stage.value} → {dc.stage.value} "
              f"(plasticity: {dc.learning_plasticity:.2f})")
        last_stage = dc.stage

print(f"\n  Final stage: {dc.stage.value}")
print(f"  Transitions: {len(dc.get_status().transitions)}")
print()

# ---------------------------------------------------------------------------
# 3. Show critical periods
# ---------------------------------------------------------------------------

print("=== Critical Periods ===")
# Reset to check at different stages
t2 = Telomere(max_operations=100)
t2.start()
dc2 = DevelopmentController(
    telomere=t2,
    critical_periods=(
        CriticalPeriod("rapid_learning", DevelopmentalStage.EMBRYONIC, DevelopmentalStage.JUVENILE, "fast template adoption"),
    ),
)

print(f"  At EMBRYONIC: rapid_learning open = {dc2.is_critical_period_open('rapid_learning')}")
for _ in range(15):
    dc2.tick()
print(f"  At JUVENILE:  rapid_learning open = {dc2.is_critical_period_open('rapid_learning')}")
print(f"  Closed periods: {[p.name for p in dc2.closed_critical_periods()]}")
print()

# ---------------------------------------------------------------------------
# 4. Capability gating
# ---------------------------------------------------------------------------

print("=== Capability Gating ===")
t3 = Telomere(max_operations=100)
t3.start()
dc3 = DevelopmentController(telomere=t3)

print(f"  Can acquire EMBRYONIC tool: {dc3.can_acquire_stage(DevelopmentalStage.EMBRYONIC)}")
print(f"  Can acquire MATURE tool:    {dc3.can_acquire_stage(DevelopmentalStage.MATURE)}")

for _ in range(75):
    dc3.tick()
print(f"  After 75 ticks ({dc3.stage.value}):")
print(f"  Can acquire MATURE tool:    {dc3.can_acquire_stage(DevelopmentalStage.MATURE)}")

# ---------------------------------------------------------------------------
# --test
# ---------------------------------------------------------------------------

assert dc.stage == DevelopmentalStage.MATURE
assert len(dc.get_status().transitions) >= 3
assert dc2.is_critical_period_open("rapid_learning") is False
assert dc3.can_acquire_stage(DevelopmentalStage.MATURE) is True
print("\n--- all assertions passed ---")
