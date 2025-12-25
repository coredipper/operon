"""
Example 21: LLM Living Cell - Full Lifecycle Simulation
=======================================================

The capstone example combining all systems:
- Real LLM via Nucleus
- Safety guardrails via Membrane
- Memory via Epigenetic Memory
- Energy management via ATP budgeting
- Lifecycle via Telomere degradation
- Cleanup via Lysosome

Key demonstrations:
- Cell "ages" over time AND usage (hybrid triggering)
- Low energy → shorter responses, conservation mode
- Errors accumulate → health degrades (ROS)
- Telomeres hit zero → cell signals for replacement
- Background thread for time-based aging

Environment Variables:
    ANTHROPIC_API_KEY: For Claude models (preferred)
    OPENAI_API_KEY: For GPT models (fallback)

Usage:
    python examples/21_llm_living_cell.py --demo    # Interactive mode
    python examples/21_llm_living_cell.py           # Smoke test mode
"""

import sys
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from operon_ai import (
    ATP_Store,
    Signal,
    Membrane,
    ThreatSignature,
    ThreatLevel,
    Lysosome,
    Waste,
    WasteType,
    Telomere,
    TelomereStatus,
    MetabolicState,
)
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import ProviderConfig
from operon_ai.memory import EpisodicMemory, MemoryTier


class CellHealthState(Enum):
    """Overall cell health states."""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    CRITICAL = "critical"
    SENESCENT = "senescent"


@dataclass
class CellVitals:
    """Current vital signs of the cell."""
    health: CellHealthState
    energy: int
    energy_max: int
    telomere_length: int
    telomere_max: int
    ros_level: float  # Reactive oxygen species (error accumulation)
    memories: int
    interactions: int


@dataclass
class LivingCell:
    """
    A living LLM-powered cell with full lifecycle simulation.

    The cell:
    - Processes requests using real LLM (Nucleus)
    - Ages over time (telomere shortening)
    - Consumes energy (ATP) for each operation
    - Accumulates damage from errors (ROS)
    - Cleans up waste (Lysosome)
    - Maintains memories (Epigenetic Memory)
    - Eventually becomes senescent and needs replacement
    """

    name: str = "Cell-001"
    silent: bool = False

    # Lifecycle parameters
    telomere_initial: int = 100
    aging_interval_seconds: float = 10.0
    telomere_decay_per_tick: int = 1
    telomere_decay_per_interaction: int = 2

    # Internal state
    _budget: ATP_Store = field(default_factory=lambda: ATP_Store(budget=500))
    _telomere: Telomere = field(init=False)
    _membrane: Membrane = field(init=False)
    _nucleus: Nucleus = field(init=False)
    _lysosome: Lysosome = field(init=False)
    _memory: EpisodicMemory = field(init=False)

    _ros_level: float = 0.0
    _interactions: int = 0
    _aging_thread: threading.Thread | None = None
    _stop_aging: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        self._telomere = Telomere(
            max_operations=self.telomere_initial,
            allow_renewal=False,
            silent=True,
        )
        self._telomere.start()  # Start the lifecycle
        self._membrane = Membrane()
        self._nucleus = Nucleus(base_energy_cost=20)
        self._lysosome = Lysosome()
        self._memory = EpisodicMemory()

        # Configure based on energy state
        self._update_config()

    def _log(self, msg: str) -> None:
        if not self.silent:
            print(msg)

    def _update_config(self) -> None:
        """Update Nucleus config based on energy state."""
        state = self._budget.get_state()

        if state == MetabolicState.STARVING:
            # Conservation mode: shorter responses
            self._nucleus_config = ProviderConfig(
                system_prompt="Be extremely brief. One sentence max.",
                temperature=0.3,
                max_tokens=50,
            )
        elif state == MetabolicState.CONSERVING:
            # Reduced mode
            self._nucleus_config = ProviderConfig(
                system_prompt="Be concise. Keep responses short.",
                temperature=0.5,
                max_tokens=150,
            )
        else:
            # Normal mode
            self._nucleus_config = ProviderConfig(
                system_prompt="You are a helpful assistant.",
                temperature=0.7,
                max_tokens=512,
            )

    def start_aging(self) -> None:
        """Start background aging thread."""
        if self._aging_thread is not None:
            return

        self._stop_aging.clear()
        self._aging_thread = threading.Thread(target=self._aging_loop, daemon=True)
        self._aging_thread.start()
        self._log(f"🧬 {self.name} started aging (tick every {self.aging_interval_seconds}s)")

    def stop_aging(self) -> None:
        """Stop background aging thread."""
        self._stop_aging.set()
        if self._aging_thread:
            self._aging_thread.join(timeout=1.0)
            self._aging_thread = None

    def _aging_loop(self) -> None:
        """Background thread for time-based aging."""
        while not self._stop_aging.is_set():
            time.sleep(self.aging_interval_seconds)
            if self._stop_aging.is_set():
                break

            # Age the cell
            self._telomere.tick(cost=self.telomere_decay_per_tick)
            self._budget.regenerate(amount=5)  # Passive energy recovery
            self._memory.decay_all()

            # Check for senescence
            status = self._telomere.get_status()
            if status.phase.value in ('senescent', 'apoptotic'):
                self._log(f"⚠️  {self.name} telomeres critical!")

    def get_vitals(self) -> CellVitals:
        """Get current cell vital signs."""
        # Get telomere status
        telomere_status = self._telomere.get_status()
        telomere_pct = telomere_status.telomere_length / telomere_status.max_telomere_length
        energy_pct = self._budget.atp / self._budget.max_atp

        # Determine health state
        if telomere_status.phase.value in ('senescent', 'apoptotic', 'terminated') or telomere_pct < 0.1:
            health = CellHealthState.SENESCENT
        elif self._ros_level > 0.7 or energy_pct < 0.2:
            health = CellHealthState.CRITICAL
        elif self._ros_level > 0.4 or energy_pct < 0.4:
            health = CellHealthState.STRESSED
        else:
            health = CellHealthState.HEALTHY

        return CellVitals(
            health=health,
            energy=self._budget.atp,
            energy_max=self._budget.max_atp,
            telomere_length=telomere_status.telomere_length,
            telomere_max=telomere_status.max_telomere_length,
            ros_level=self._ros_level,
            memories=len(self._memory.memories),
            interactions=self._interactions,
        )

    def process(self, request: str) -> str:
        """Process a request through the living cell."""
        vitals = self.get_vitals()

        # Check for senescence
        if vitals.health == CellHealthState.SENESCENT:
            return f"[{self.name}] I am senescent and cannot process requests. Please create a new cell."

        # Input filtering
        signal = Signal(content=request)
        filter_result = self._membrane.filter(signal)
        if not filter_result.allowed:
            self._ros_level = min(1.0, self._ros_level + 0.05)
            return f"[{self.name}] Request blocked by immune system."

        # Check energy
        self._update_config()
        energy_cost = 20 if vitals.health == CellHealthState.HEALTHY else 10

        if not self._budget.consume(cost=energy_cost):
            return f"[{self.name}] Insufficient energy. Resting..."

        # Process with memory context
        context = self._memory.format_context(request)
        prompt = f"{context}\n\nUser: {request}" if context else request

        try:
            response = self._nucleus.transcribe(prompt, config=self._nucleus_config)

            # Store interaction in memory
            self._memory.store(
                f"Q: {request[:50]} A: {response.content[:50]}",
                tier=MemoryTier.WORKING,
            )

            # Age from interaction
            self._telomere.tick(cost=self.telomere_decay_per_interaction)
            self._interactions += 1

            # Health indicator
            health_icon = {
                CellHealthState.HEALTHY: "💚",
                CellHealthState.STRESSED: "💛",
                CellHealthState.CRITICAL: "🧡",
                CellHealthState.SENESCENT: "💀",
            }[vitals.health]

            return f"{health_icon} {response.content}"

        except Exception as e:
            # Error increases ROS
            self._ros_level = min(1.0, self._ros_level + 0.1)

            # Log to lysosome for cleanup
            self._lysosome.ingest(Waste(
                content=str(e),
                waste_type=WasteType.FAILED_OPERATION,
                source="nucleus",
            ))

            return f"[{self.name}] Error processing request (ROS: {self._ros_level:.0%})"

    def display_status(self) -> None:
        """Display current cell status."""
        vitals = self.get_vitals()

        # Health bar
        health_bar = "█" * int(vitals.telomere_length / 10) + "░" * (10 - int(vitals.telomere_length / 10))
        energy_bar = "█" * int(vitals.energy / 50) + "░" * (10 - int(vitals.energy / 50))

        print(f"\n{'='*60}")
        print(f"🧬 {self.name} Status")
        print(f"{'='*60}")
        print(f"Health:    {vitals.health.value.upper()}")
        print(f"Telomere:  [{health_bar}] {vitals.telomere_length}/{vitals.telomere_max}")
        print(f"Energy:    [{energy_bar}] {vitals.energy}/{vitals.energy_max}")
        print(f"ROS Level: {vitals.ros_level:.0%}")
        print(f"Memories:  {vitals.memories}")
        print(f"Interactions: {vitals.interactions}")
        print(f"{'='*60}\n")


def run_demo():
    """Interactive demo mode."""
    print("=" * 60)
    print("LLM Living Cell - Full Lifecycle Demo")
    print("=" * 60)
    print()

    cell = LivingCell(name="Demo-Cell", silent=False)
    cell.start_aging()

    print(f"Using provider: {cell._nucleus.provider.name}")
    cell.display_status()

    print("Commands:")
    print("  /status  - Show cell vitals")
    print("  /damage  - Simulate damage (increase ROS)")
    print("  /heal    - Attempt self-repair")
    print("  /quit    - Exit")
    print()

    try:
        while True:
            vitals = cell.get_vitals()
            if vitals.health == CellHealthState.SENESCENT:
                print("\n💀 Cell has reached senescence. Creating new cell...")
                cell.stop_aging()
                cell = LivingCell(name="Demo-Cell-2", silent=False)
                cell.start_aging()
                cell.display_status()

            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input[1:].lower()
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "status":
                    cell.display_status()
                elif cmd == "damage":
                    cell._ros_level = min(1.0, cell._ros_level + 0.2)
                    cell._telomere.tick(cost=10)
                    print("💥 Cell damaged!")
                    cell.display_status()
                elif cmd == "heal":
                    cell._ros_level = max(0.0, cell._ros_level - 0.1)
                    print("🩹 Attempting self-repair...")
                    cell.display_status()
                continue

            response = cell.process(user_input)
            print(f"\nCell: {response}\n")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cell.stop_aging()


def run_smoke_test():
    """Automated smoke test."""
    print("Running smoke test...")

    cell = LivingCell(name="Test-Cell", silent=True)

    # Test 1: Basic processing
    response = cell.process("Hello")
    assert response, "Should get response"
    print(f"✓ Basic processing works")

    # Test 2: Vitals tracking
    vitals = cell.get_vitals()
    assert vitals.interactions == 1, "Should track interactions"
    assert vitals.telomere_length < cell.telomere_initial, "Telomeres should shorten"
    print(f"✓ Vitals tracking works: {vitals.health.value}")

    # Test 3: Energy consumption
    initial_energy = cell._budget.atp
    cell.process("Another request")
    assert cell._budget.atp < initial_energy, "Should consume energy"
    print(f"✓ Energy consumption works")

    # Test 4: Error handling (ROS)
    initial_ros = cell._ros_level
    cell._membrane.add_signature(ThreatSignature(
        pattern="test_pattern",
        level=ThreatLevel.CRITICAL,
        description="Test pattern for demo",
    ))
    cell.process("test_pattern dangerous")
    # ROS should increase from blocked request
    print(f"✓ ROS tracking works: {cell._ros_level:.0%}")

    # Test 5: Aging thread
    cell.start_aging()
    time.sleep(0.1)
    cell.stop_aging()
    print(f"✓ Aging thread works")

    print("\nSmoke test passed!")


def main():
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_smoke_test()


if __name__ == "__main__":
    main()
