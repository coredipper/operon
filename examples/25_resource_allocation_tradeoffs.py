#!/usr/bin/env python3
"""
Example 25: Resource Allocation Trade-offs
==========================================

Based on concepts from "Resource Allocation in Mammalian Systems" (Baghdassarian & Lewis, 2024).

Demonstrates:
- A coupled budget of nutrients, machinery, and bioenergy.
- Trade-offs between growth (translation), maintenance, and specialization.
- Dynamic reallocation under nutrient scarcity to preserve ATP.
- Autophagy-style recycling when energy is critically low.

Run:
    python examples/25_resource_allocation_tradeoffs.py
"""

from __future__ import annotations

from dataclasses import dataclass

from operon_ai import (
    ATP_Store,
    Lysosome,
    Mitochondria,
    NegativeFeedbackLoop,
    Waste,
    WasteType,
)


@dataclass
class Scenario:
    name: str
    nutrient_supply: float
    cycles: int = 3


@dataclass
class ProteomeAllocation:
    translation: float
    energy: float
    maintenance: float
    specialization: float

    def normalize(self) -> None:
        total = self.translation + self.energy + self.maintenance + self.specialization
        if total == 0:
            return
        self.translation /= total
        self.energy /= total
        self.maintenance /= total
        self.specialization /= total


MACHINERY_CAPACITY = 100.0
ENERGY_EFFICIENCY = 2.0
MAINTENANCE_COST = 18
TRANSLATION_COST_PER_UNIT = 0.9
SPECIALIZATION_COST_PER_UNIT = 1.2


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def base_allocation(nutrient_supply: float) -> ProteomeAllocation:
    if nutrient_supply < 60:
        allocation = ProteomeAllocation(
            translation=0.25,
            energy=0.30,
            maintenance=0.35,
            specialization=0.10,
        )
    elif nutrient_supply < 100:
        allocation = ProteomeAllocation(
            translation=0.35,
            energy=0.25,
            maintenance=0.28,
            specialization=0.12,
        )
    else:
        allocation = ProteomeAllocation(
            translation=0.42,
            energy=0.18,
            maintenance=0.22,
            specialization=0.18,
        )
    allocation.normalize()
    return allocation


def recycle_reserve(waste: Waste) -> dict:
    payload = waste.content if isinstance(waste.content, dict) else {}
    return {"atp": int(payload.get("atp", 0))}


def adjust_allocation(
    allocation: ProteomeAllocation,
    correction: float,
    min_translation: float = 0.1,
    max_energy: float = 0.6,
) -> None:
    energy = clamp(allocation.energy + correction, 0.15, max_energy)
    translation = 1.0 - (energy + allocation.maintenance + allocation.specialization)

    if translation < min_translation:
        translation = min_translation
        energy = 1.0 - (translation + allocation.maintenance + allocation.specialization)

    allocation.energy = energy
    allocation.translation = translation


def simulate_scenario(scenario: Scenario) -> None:
    print("\n" + "=" * 70)
    print(f"Scenario: {scenario.name} (nutrient supply = {scenario.nutrient_supply})")
    print("=" * 70)

    allocation = base_allocation(scenario.nutrient_supply)

    energy_store = ATP_Store(budget=int(scenario.nutrient_supply), silent=True)
    mito = Mitochondria(silent=True)
    feedback = NegativeFeedbackLoop(
        setpoint=1.0,
        gain=0.4,
        damping=0.1,
        min_correction=0.02,
        max_correction=0.2,
        silent=True,
    )
    lysosome = Lysosome(
        digesters={WasteType.ORPHANED_RESOURCE: recycle_reserve},
        silent=True,
    )

    deficit_streak = 0

    for cycle in range(1, scenario.cycles + 1):
        print(f"\nCycle {cycle}")
        print("-" * 60)

        expression = f"{scenario.nutrient_supply} * {allocation.energy} * {ENERGY_EFFICIENCY}"
        energy_result = mito.metabolize(expression)
        produced = int(energy_result.atp.value) if energy_result.success and energy_result.atp else 0
        energy_store.regenerate(produced)

        translation_units = MACHINERY_CAPACITY * allocation.translation
        specialization_units = MACHINERY_CAPACITY * allocation.specialization
        translation_cost = int(translation_units * TRANSLATION_COST_PER_UNIT)
        specialization_cost = int(specialization_units * SPECIALIZATION_COST_PER_UNIT)

        total_demand = MAINTENANCE_COST + translation_cost + specialization_cost
        available = energy_store.atp
        gap = available - total_demand
        coverage = available / max(1, total_demand)

        print(
            f"Budget -> ATP: {available:3d}, Demand: {total_demand:3d}, "
            f"Gap: {gap:+d}, Coverage: {coverage:.2f}x"
        )
        print(
            "Allocation -> "
            f"translation={allocation.translation:.2f}, "
            f"energy={allocation.energy:.2f}, "
            f"maintenance={allocation.maintenance:.2f}, "
            f"specialization={allocation.specialization:.2f}"
        )

        deferred = []

        if not energy_store.consume(MAINTENANCE_COST, "maintenance", priority=10):
            deferred.append("maintenance")
        if not energy_store.consume(translation_cost, "translation", priority=5):
            deferred.append("translation")
        if not energy_store.consume(specialization_cost, "specialization", priority=1):
            deferred.append("specialization")

        state = energy_store.get_state()

        if coverage < 1.0:
            deficit_streak += 1
        else:
            deficit_streak = 0

        energy_ratio = energy_store.atp / max(1, energy_store.max_atp)
        if deficit_streak >= 2 and energy_ratio < 0.3:
            lysosome.ingest(
                Waste(
                    waste_type=WasteType.ORPHANED_RESOURCE,
                    content={"atp": 25},
                    source="autophagy",
                )
            )
            recovered = lysosome.digest().recycled.get("atp", 0)
            if recovered:
                energy_store.regenerate(int(recovered))
                print(f"Autophagy recovered {recovered} ATP")

        if deferred:
            print("Deferred objectives:", ", ".join(deferred))

        print(f"Energy state: {state.value} ({energy_store.atp}/{energy_store.max_atp} ATP)")

        correction = feedback.measure(coverage)
        adjust_allocation(allocation, correction)


def main() -> None:
    scenarios = [
        Scenario(name="Scarcity", nutrient_supply=45),
        Scenario(name="Baseline", nutrient_supply=85),
        Scenario(name="Abundance", nutrient_supply=130),
    ]

    print("Resource Allocation Trade-offs")
    print("=" * 70)

    for scenario in scenarios:
        simulate_scenario(scenario)


if __name__ == "__main__":
    main()
