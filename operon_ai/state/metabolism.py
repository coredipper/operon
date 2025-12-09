"""
Metabolism: Cellular Energy Management System
=============================================

Biological Analogy:
- ATP: Primary energy currency (like tokens/compute)
- GTP: Secondary currency for specific processes (like premium features)
- NADH: Electron carrier / energy reserve (like cached results)
- Glycolysis: Fast energy from simple sources
- Oxidative phosphorylation: Slow but efficient energy production
- Metabolic rate: How fast energy is consumed/regenerated
- Starvation response: Behavior when energy is critically low

The metabolism system manages all energy resources for an agent,
preventing resource exhaustion (ischemia) and enabling graceful
degradation under resource pressure.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import threading
import time


class MetabolicState(Enum):
    """Current metabolic state of the cell."""
    NORMAL = "normal"           # Plenty of energy
    CONSERVING = "conserving"   # Low energy, reducing activity
    STARVING = "starving"       # Critical, survival mode only
    FEASTING = "feasting"       # Excess energy, can do extra work
    DORMANT = "dormant"         # Minimal activity, maximum conservation


class EnergyType(Enum):
    """Types of cellular energy currencies."""
    ATP = "atp"       # Primary currency - general operations
    GTP = "gtp"       # Secondary - specialized operations (like tool calls)
    NADH = "nadh"     # Reserve - can be converted to ATP


@dataclass
class EnergyTransaction:
    """Record of an energy transaction."""
    energy_type: EnergyType
    amount: int
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True


@dataclass
class MetabolicReport:
    """Comprehensive metabolic status report."""
    state: MetabolicState
    atp: int
    gtp: int
    nadh: int
    total_capacity: int
    utilization: float
    regeneration_rate: float
    transactions_count: int
    debt: int
    health_score: float


class ATP_Store:
    """
    Advanced Metabolic Budget Manager.

    Manages multiple energy currencies with regeneration, debt tracking,
    and adaptive behavior based on resource availability.

    Features:

    1. Multiple Energy Currencies
       - ATP: General operations (default)
       - GTP: Premium/specialized operations
       - NADH: Reserve that converts to ATP

    2. Regeneration
       - Passive regeneration over time
       - Configurable regeneration rate
       - NADH → ATP conversion

    3. Metabolic States
       - NORMAL: Full operation
       - CONSERVING: Reduced activity
       - STARVING: Survival mode
       - FEASTING: Excess capacity
       - DORMANT: Minimal operation

    4. Energy Debt
       - Can go into debt for critical operations
       - Debt must be repaid before normal operation
       - Interest on outstanding debt

    5. Energy Sharing
       - Transfer energy between agents
       - Pool energy for colony operations

    Example:
        >>> store = ATP_Store(budget=100)
        >>> store.consume(10, "query_llm")
        True
        >>> store.get_balance()
        90
        >>> store.regenerate(5)
        >>> store.get_balance()
        95
    """

    # Thresholds for metabolic states
    CONSERVING_THRESHOLD = 0.3   # Below 30% = conserving
    STARVING_THRESHOLD = 0.1    # Below 10% = starving
    FEASTING_THRESHOLD = 0.9    # Above 90% = feasting

    def __init__(
        self,
        budget: int,
        gtp_budget: int = 0,
        nadh_reserve: int = 0,
        regeneration_rate: float = 0.0,
        max_debt: int = 0,
        debt_interest: float = 0.1,
        on_state_change: Callable[[MetabolicState], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the ATP Store.

        Args:
            budget: Initial ATP budget
            gtp_budget: Initial GTP budget (premium operations)
            nadh_reserve: Initial NADH reserve (convertible to ATP)
            regeneration_rate: ATP regenerated per second (0 = disabled)
            max_debt: Maximum allowed energy debt
            debt_interest: Interest rate on debt (per operation)
            on_state_change: Callback when metabolic state changes
            silent: Suppress console output
        """
        self.atp = budget
        self.max_atp = budget
        self.gtp = gtp_budget
        self.max_gtp = gtp_budget
        self.nadh = nadh_reserve
        self.max_nadh = nadh_reserve

        self.regeneration_rate = regeneration_rate
        self.max_debt = max_debt
        self.debt_interest = debt_interest
        self.on_state_change = on_state_change
        self.silent = silent

        # Current state
        self._debt = 0
        self._state = MetabolicState.NORMAL
        self._last_regeneration = datetime.now()
        self._transactions: list[EnergyTransaction] = []
        self._lock = threading.Lock()

        # Statistics
        self._total_consumed = 0
        self._total_regenerated = 0
        self._operations_count = 0
        self._failed_operations = 0

        # Start regeneration thread if rate > 0
        self._regeneration_thread: threading.Thread | None = None
        self._stop_regeneration = threading.Event()
        if regeneration_rate > 0:
            self._start_regeneration()

    def _start_regeneration(self):
        """Start background regeneration thread."""
        def regenerate_loop():
            while not self._stop_regeneration.is_set():
                time.sleep(1.0)  # Regenerate every second
                if not self._stop_regeneration.is_set():
                    self.regenerate(int(self.regeneration_rate))

        self._regeneration_thread = threading.Thread(target=regenerate_loop, daemon=True)
        self._regeneration_thread.start()

    def stop_regeneration(self):
        """Stop the background regeneration thread."""
        self._stop_regeneration.set()
        if self._regeneration_thread:
            self._regeneration_thread.join(timeout=2.0)

    def consume(
        self,
        cost: int,
        operation: str = "unknown",
        energy_type: EnergyType = EnergyType.ATP,
        allow_debt: bool = False,
        priority: int = 0,
    ) -> bool:
        """
        Consume energy for an operation.

        Args:
            cost: Amount of energy to consume
            operation: Description of the operation
            energy_type: Which energy currency to use
            allow_debt: Whether to allow going into debt
            priority: Higher priority operations can use debt

        Returns:
            True if energy was available, False otherwise
        """
        with self._lock:
            self._operations_count += 1

            # Check metabolic state - may reject non-critical operations
            if self._state == MetabolicState.STARVING and priority < 5:
                if not self.silent:
                    print("💀 [Metabolism] STARVING: Only critical operations allowed")
                self._failed_operations += 1
                return False

            if self._state == MetabolicState.DORMANT and priority < 10:
                if not self.silent:
                    print("😴 [Metabolism] DORMANT: Minimal operations only")
                self._failed_operations += 1
                return False

            # Select energy pool
            if energy_type == EnergyType.ATP:
                balance = self.atp
            elif energy_type == EnergyType.GTP:
                balance = self.gtp
            else:
                balance = self.nadh

            # Check if we have enough
            if balance >= cost:
                # Simple deduction
                if energy_type == EnergyType.ATP:
                    self.atp -= cost
                elif energy_type == EnergyType.GTP:
                    self.gtp -= cost
                else:
                    self.nadh -= cost

                self._total_consumed += cost
                self._record_transaction(energy_type, -cost, operation, True)
                self._update_state()
                return True

            # Try to use NADH reserve for ATP
            if energy_type == EnergyType.ATP and self.nadh > 0:
                conversion = min(self.nadh, cost - balance)
                self.nadh -= conversion
                self.atp += conversion
                if not self.silent:
                    print(f"🔄 [Metabolism] Converted {conversion} NADH → ATP")

                if self.atp >= cost:
                    self.atp -= cost
                    self._total_consumed += cost
                    self._record_transaction(energy_type, -cost, operation, True)
                    self._update_state()
                    return True

            # Try to use debt
            if allow_debt and self._debt < self.max_debt:
                deficit = cost - balance
                if self._debt + deficit <= self.max_debt:
                    # Take on debt
                    self._debt += deficit
                    if energy_type == EnergyType.ATP:
                        self.atp = 0
                    elif energy_type == EnergyType.GTP:
                        self.gtp = 0

                    if not self.silent:
                        print(f"💳 [Metabolism] Energy debt: +{deficit} (total: {self._debt})")

                    self._total_consumed += cost
                    self._record_transaction(energy_type, -cost, operation, True)
                    self._update_state()
                    return True

            # Cannot afford operation
            if not self.silent:
                print(f"💀 [Metabolism] APOPTOSIS WARNING: Insufficient {energy_type.value} for {operation}")
            self._failed_operations += 1
            self._record_transaction(energy_type, 0, operation, False)
            return False

    def regenerate(self, amount: int, energy_type: EnergyType = EnergyType.ATP):
        """
        Regenerate energy.

        First pays off debt, then adds to balance.
        """
        with self._lock:
            remaining = amount

            # Pay off debt first
            if self._debt > 0 and energy_type == EnergyType.ATP:
                debt_payment = min(self._debt, remaining)
                self._debt -= debt_payment
                remaining -= debt_payment
                if debt_payment > 0 and not self.silent:
                    print(f"💰 [Metabolism] Paid {debt_payment} debt (remaining: {self._debt})")

            # Add to balance
            if remaining > 0:
                if energy_type == EnergyType.ATP:
                    self.atp = min(self.max_atp, self.atp + remaining)
                elif energy_type == EnergyType.GTP:
                    self.gtp = min(self.max_gtp, self.gtp + remaining)
                else:
                    self.nadh = min(self.max_nadh, self.nadh + remaining)

                self._total_regenerated += remaining

            self._update_state()

    def transfer_to(self, other: 'ATP_Store', amount: int, energy_type: EnergyType = EnergyType.ATP) -> bool:
        """
        Transfer energy to another store.

        Enables energy sharing between agents in a colony.
        """
        with self._lock:
            if energy_type == EnergyType.ATP:
                if self.atp < amount:
                    return False
                self.atp -= amount
            elif energy_type == EnergyType.GTP:
                if self.gtp < amount:
                    return False
                self.gtp -= amount
            else:
                if self.nadh < amount:
                    return False
                self.nadh -= amount

        other.regenerate(amount, energy_type)

        if not self.silent:
            print(f"🔀 [Metabolism] Transferred {amount} {energy_type.value}")

        return True

    def convert_nadh_to_atp(self, amount: int) -> int:
        """
        Convert NADH reserve to ATP.

        Biological analogy: oxidative phosphorylation.
        Returns actual amount converted.
        """
        with self._lock:
            conversion = min(amount, self.nadh, self.max_atp - self.atp)
            if conversion > 0:
                self.nadh -= conversion
                self.atp += conversion
                if not self.silent:
                    print(f"🔄 [Metabolism] Oxidative phosphorylation: {conversion} NADH → ATP")
            return conversion

    def _update_state(self):
        """Update metabolic state based on current energy levels."""
        old_state = self._state

        total_capacity = self.max_atp + self.max_gtp
        total_current = self.atp + self.gtp

        if total_capacity == 0:
            ratio = 0.0
        else:
            ratio = total_current / total_capacity

        # Account for debt
        if self._debt > 0:
            ratio -= (self._debt / total_capacity) * 0.5

        if ratio <= self.STARVING_THRESHOLD:
            self._state = MetabolicState.STARVING
        elif ratio <= self.CONSERVING_THRESHOLD:
            self._state = MetabolicState.CONSERVING
        elif ratio >= self.FEASTING_THRESHOLD:
            self._state = MetabolicState.FEASTING
        else:
            self._state = MetabolicState.NORMAL

        if old_state != self._state:
            if not self.silent:
                print(f"🔋 [Metabolism] State: {old_state.value} → {self._state.value}")
            if self.on_state_change:
                self.on_state_change(self._state)

    def _record_transaction(self, energy_type: EnergyType, amount: int, operation: str, success: bool):
        """Record a transaction for audit."""
        self._transactions.append(EnergyTransaction(
            energy_type=energy_type,
            amount=amount,
            operation=operation,
            success=success
        ))

        # Keep only last 1000 transactions
        if len(self._transactions) > 1000:
            self._transactions = self._transactions[-1000:]

    def apply_debt_interest(self):
        """Apply interest to outstanding debt."""
        if self._debt > 0:
            interest = int(self._debt * self.debt_interest)
            self._debt += interest
            if not self.silent:
                print(f"💸 [Metabolism] Debt interest: +{interest} (total: {self._debt})")

    def enter_dormancy(self):
        """Enter dormant state for maximum energy conservation."""
        with self._lock:
            self._state = MetabolicState.DORMANT
            if not self.silent:
                print("😴 [Metabolism] Entering dormancy...")

    def exit_dormancy(self):
        """Exit dormant state."""
        with self._lock:
            self._update_state()
            if not self.silent:
                print("☀️ [Metabolism] Exiting dormancy...")

    def get_balance(self, energy_type: EnergyType = EnergyType.ATP) -> int:
        """Get current balance of specified energy type."""
        if energy_type == EnergyType.ATP:
            return self.atp
        elif energy_type == EnergyType.GTP:
            return self.gtp
        return self.nadh

    def get_state(self) -> MetabolicState:
        """Get current metabolic state."""
        return self._state

    def get_debt(self) -> int:
        """Get current energy debt."""
        return self._debt

    def get_report(self) -> MetabolicReport:
        """Get comprehensive metabolic report."""
        total_capacity = self.max_atp + self.max_gtp + self.max_nadh
        total_current = self.atp + self.gtp + self.nadh

        # Calculate health score (0-1)
        utilization = total_current / max(1, total_capacity)
        debt_penalty = self._debt / max(1, total_capacity)
        failure_rate = self._failed_operations / max(1, self._operations_count)
        health = max(0, min(1, utilization - debt_penalty - (failure_rate * 0.5)))

        return MetabolicReport(
            state=self._state,
            atp=self.atp,
            gtp=self.gtp,
            nadh=self.nadh,
            total_capacity=total_capacity,
            utilization=utilization,
            regeneration_rate=self.regeneration_rate,
            transactions_count=len(self._transactions),
            debt=self._debt,
            health_score=health
        )

    def get_statistics(self) -> dict:
        """Get metabolic statistics."""
        return {
            "atp": self.atp,
            "max_atp": self.max_atp,
            "gtp": self.gtp,
            "max_gtp": self.max_gtp,
            "nadh": self.nadh,
            "max_nadh": self.max_nadh,
            "state": self._state.value,
            "debt": self._debt,
            "total_consumed": self._total_consumed,
            "total_regenerated": self._total_regenerated,
            "operations_count": self._operations_count,
            "failed_operations": self._failed_operations,
            "success_rate": 1 - (self._failed_operations / max(1, self._operations_count)),
        }

    def get_transactions(self, limit: int = 100) -> list[EnergyTransaction]:
        """Get recent transactions."""
        return self._transactions[-limit:]

    def reset(self):
        """Reset to initial state."""
        with self._lock:
            self.atp = self.max_atp
            self.gtp = self.max_gtp
            self.nadh = self.max_nadh
            self._debt = 0
            self._transactions.clear()
            self._total_consumed = 0
            self._total_regenerated = 0
            self._operations_count = 0
            self._failed_operations = 0
            self._update_state()

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_regeneration()
