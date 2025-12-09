"""
Telomere: Lifecycle and Senescence Management
==============================================

Biological Analogy:
- Telomeres: Protective caps on chromosomes that shorten with each division
- Hayflick limit: Maximum number of cell divisions before senescence
- Senescence: Cellular aging leading to reduced function
- Apoptosis: Programmed cell death when telomeres are depleted
- Telomerase: Enzyme that can extend telomeres (rejuvenation)
- Cellular clock: Tracking of cell age and remaining lifespan

The Telomere system manages agent lifecycle, tracking usage,
detecting when agents should be retired, and enabling graceful
shutdown or renewal.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import threading


class LifecyclePhase(Enum):
    """Current phase in the agent's lifecycle."""
    NASCENT = "nascent"           # Just created, initializing
    ACTIVE = "active"             # Normal operation
    SENESCENT = "senescent"       # Aging, reduced capability
    APOPTOTIC = "apoptotic"       # Preparing for shutdown
    TERMINATED = "terminated"      # No longer operational


class SenescenceReason(Enum):
    """Why an agent entered senescence."""
    TELOMERE_DEPLETION = "telomere_depletion"  # Natural aging
    ERROR_ACCUMULATION = "error_accumulation"   # Too many errors
    RESOURCE_EXHAUSTION = "resource_exhaustion" # Out of resources
    MANUAL_TRIGGER = "manual_trigger"           # Explicitly triggered
    TIMEOUT = "timeout"                         # Exceeded time limit
    IDLE_TIMEOUT = "idle_timeout"               # Inactive too long


@dataclass
class LifecycleEvent:
    """Record of a lifecycle event."""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


@dataclass
class TelomereStatus:
    """Current telomere/lifecycle status."""
    phase: LifecyclePhase
    telomere_length: int
    max_telomere_length: int
    operations_remaining: int
    time_remaining: timedelta | None
    health_score: float
    senescence_reason: SenescenceReason | None


class Telomere:
    """
    Lifecycle and Senescence Manager.

    Tracks agent lifespan and triggers appropriate lifecycle
    transitions based on usage, errors, and time.

    Features:

    1. Operation Counting (Telomere Shortening)
       - Each operation shortens telomeres
       - When telomeres depleted, enter senescence
       - Configurable Hayflick limit

    2. Time-Based Aging
       - Maximum lifetime for agents
       - Idle timeout detection
       - Grace periods before termination

    3. Error-Based Senescence
       - Track error accumulation
       - Trigger senescence above threshold
       - Recovery mechanisms

    4. Lifecycle Phases
       - NASCENT: Initialization
       - ACTIVE: Normal operation
       - SENESCENT: Degraded capability
       - APOPTOTIC: Preparing for shutdown
       - TERMINATED: No longer operational

    5. Renewal (Telomerase)
       - Extend telomeres (rejuvenation)
       - Reset error counts
       - Requires explicit permission

    6. Callbacks
       - Notify on phase transitions
       - Enable cleanup and handoff

    Example:
        >>> telomere = Telomere(max_operations=1000, max_lifetime_hours=24)
        >>> telomere.start()
        >>> for i in range(100):
        ...     if telomere.tick():  # Returns False when senescent
        ...         do_operation()
        >>> print(telomere.get_status().operations_remaining)
        900
    """

    # Thresholds for lifecycle transitions
    SENESCENCE_THRESHOLD = 0.1   # Enter senescence below 10% telomere
    WARNING_THRESHOLD = 0.2      # Warn below 20%
    ERROR_SENESCENCE_RATE = 0.5  # Error ratio that triggers senescence

    def __init__(
        self,
        max_operations: int = 10000,
        max_lifetime_hours: float | None = None,
        idle_timeout_minutes: float | None = None,
        error_threshold: int = 100,
        allow_renewal: bool = True,
        on_phase_change: Callable[[LifecyclePhase, LifecyclePhase], None] | None = None,
        on_senescence: Callable[[SenescenceReason], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Telomere.

        Args:
            max_operations: Maximum operations before senescence (Hayflick limit)
            max_lifetime_hours: Maximum time before senescence (None = unlimited)
            idle_timeout_minutes: Time without activity before senescence
            error_threshold: Number of errors before senescence
            allow_renewal: Whether telomere extension is allowed
            on_phase_change: Callback when lifecycle phase changes
            on_senescence: Callback when entering senescence
            silent: Suppress console output
        """
        self.max_operations = max_operations
        self.max_lifetime = timedelta(hours=max_lifetime_hours) if max_lifetime_hours else None
        self.idle_timeout = timedelta(minutes=idle_timeout_minutes) if idle_timeout_minutes else None
        self.error_threshold = error_threshold
        self.allow_renewal = allow_renewal
        self.on_phase_change = on_phase_change
        self.on_senescence = on_senescence
        self.silent = silent

        # Current state
        self._telomere_length = max_operations
        self._phase = LifecyclePhase.NASCENT
        self._senescence_reason: SenescenceReason | None = None

        # Timing
        self._created_at: datetime | None = None
        self._started_at: datetime | None = None
        self._last_activity: datetime | None = None
        self._terminated_at: datetime | None = None

        # Counters
        self._operations_count = 0
        self._error_count = 0
        self._renewal_count = 0

        # Event log
        self._events: list[LifecycleEvent] = []
        self._lock = threading.Lock()

        self._log_event("created", {"max_operations": max_operations})

    def start(self):
        """Start the agent lifecycle (transition from NASCENT to ACTIVE)."""
        with self._lock:
            if self._phase != LifecyclePhase.NASCENT:
                return

            self._started_at = datetime.now()
            self._last_activity = self._started_at
            self._transition_to(LifecyclePhase.ACTIVE)
            self._log_event("started")

    def tick(self, cost: int = 1) -> bool:
        """
        Record an operation and shorten telomeres.

        Returns True if the agent can continue operating,
        False if it has entered senescence or later phases.
        """
        with self._lock:
            # Check if already in terminal state
            if self._phase in (LifecyclePhase.APOPTOTIC, LifecyclePhase.TERMINATED):
                return False

            # Auto-start if still nascent
            if self._phase == LifecyclePhase.NASCENT:
                self.start()

            self._operations_count += 1
            self._telomere_length = max(0, self._telomere_length - cost)
            self._last_activity = datetime.now()

            # Check for various senescence triggers
            self._check_senescence()

            return self._phase == LifecyclePhase.ACTIVE

    def record_error(self) -> bool:
        """
        Record an error.

        Returns True if the agent can continue,
        False if error threshold triggered senescence.
        """
        with self._lock:
            self._error_count += 1
            self._log_event("error", {"error_count": self._error_count})

            # Check error-based senescence
            if self._error_count >= self.error_threshold:
                self._enter_senescence(SenescenceReason.ERROR_ACCUMULATION)
                return False

            # Check error rate
            if self._operations_count > 0:
                error_rate = self._error_count / self._operations_count
                if error_rate >= self.ERROR_SENESCENCE_RATE:
                    self._enter_senescence(SenescenceReason.ERROR_ACCUMULATION)
                    return False

            return self._phase == LifecyclePhase.ACTIVE

    def heartbeat(self):
        """Update last activity timestamp (prevent idle timeout)."""
        with self._lock:
            self._last_activity = datetime.now()

    def check_timeouts(self) -> bool:
        """
        Check for time-based senescence triggers.

        Should be called periodically. Returns True if still active.
        """
        with self._lock:
            if self._phase != LifecyclePhase.ACTIVE:
                return self._phase not in (LifecyclePhase.APOPTOTIC, LifecyclePhase.TERMINATED)

            now = datetime.now()

            # Check max lifetime
            if self.max_lifetime and self._started_at:
                age = now - self._started_at
                if age >= self.max_lifetime:
                    self._enter_senescence(SenescenceReason.TIMEOUT)
                    return False

            # Check idle timeout
            if self.idle_timeout and self._last_activity:
                idle_time = now - self._last_activity
                if idle_time >= self.idle_timeout:
                    self._enter_senescence(SenescenceReason.IDLE_TIMEOUT)
                    return False

            return True

    def renew(self, amount: int | None = None, reset_errors: bool = True) -> bool:
        """
        Extend telomeres (rejuvenation).

        Like telomerase in biology, this can extend the agent's lifespan.
        Requires allow_renewal=True.
        """
        if not self.allow_renewal:
            if not self.silent:
                print("🧬 [Telomere] Renewal not allowed")
            return False

        with self._lock:
            # Can't renew if already terminated
            if self._phase == LifecyclePhase.TERMINATED:
                return False

            # Restore telomere length
            amount = amount or self.max_operations
            self._telomere_length = min(self.max_operations, self._telomere_length + amount)

            # Reset errors if requested
            if reset_errors:
                self._error_count = 0

            self._renewal_count += 1
            self._log_event("renewed", {
                "amount": amount,
                "new_length": self._telomere_length,
                "errors_reset": reset_errors
            })

            # Can recover from senescence
            if self._phase == LifecyclePhase.SENESCENT:
                self._transition_to(LifecyclePhase.ACTIVE)
                self._senescence_reason = None

            if not self.silent:
                print(f"🔄 [Telomere] Renewed: {self._telomere_length}/{self.max_operations}")

            return True

    def trigger_apoptosis(self, reason: str = ""):
        """Trigger programmed cell death (graceful shutdown)."""
        with self._lock:
            if self._phase == LifecyclePhase.TERMINATED:
                return

            self._log_event("apoptosis_triggered", {"reason": reason})
            self._transition_to(LifecyclePhase.APOPTOTIC)

    def terminate(self):
        """Forcefully terminate the agent."""
        with self._lock:
            self._terminated_at = datetime.now()
            self._log_event("terminated")
            self._transition_to(LifecyclePhase.TERMINATED)

    def _check_senescence(self):
        """Check all senescence conditions."""
        # Telomere depletion
        if self._telomere_length <= 0:
            self._enter_senescence(SenescenceReason.TELOMERE_DEPLETION)
            return

        # Check warning threshold
        ratio = self._telomere_length / self.max_operations
        if ratio <= self.SENESCENCE_THRESHOLD:
            self._enter_senescence(SenescenceReason.TELOMERE_DEPLETION)
        elif ratio <= self.WARNING_THRESHOLD and not self.silent:
            print(f"⚠️ [Telomere] Warning: {int(ratio * 100)}% remaining")

    def _enter_senescence(self, reason: SenescenceReason):
        """Enter senescence state."""
        if self._phase in (LifecyclePhase.SENESCENT, LifecyclePhase.APOPTOTIC, LifecyclePhase.TERMINATED):
            return

        self._senescence_reason = reason
        self._log_event("senescence", {"reason": reason.value})
        self._transition_to(LifecyclePhase.SENESCENT)

        if self.on_senescence:
            self.on_senescence(reason)

        if not self.silent:
            print(f"🧓 [Telomere] Senescence: {reason.value}")

    def _transition_to(self, new_phase: LifecyclePhase):
        """Transition to a new lifecycle phase."""
        old_phase = self._phase
        self._phase = new_phase

        self._log_event("phase_change", {
            "from": old_phase.value,
            "to": new_phase.value
        })

        if self.on_phase_change:
            self.on_phase_change(old_phase, new_phase)

        if not self.silent:
            print(f"🔄 [Telomere] Phase: {old_phase.value} → {new_phase.value}")

    def _log_event(self, event_type: str, details: dict | None = None):
        """Log a lifecycle event."""
        self._events.append(LifecycleEvent(
            event_type=event_type,
            details=details or {}
        ))

        # Keep only last 1000 events
        if len(self._events) > 1000:
            self._events = self._events[-1000:]

    def get_status(self) -> TelomereStatus:
        """Get current telomere/lifecycle status."""
        time_remaining = None
        if self.max_lifetime and self._started_at:
            age = datetime.now() - self._started_at
            time_remaining = max(timedelta(0), self.max_lifetime - age)

        # Calculate health score
        telomere_ratio = self._telomere_length / max(1, self.max_operations)
        error_penalty = self._error_count / max(1, self.error_threshold)
        health = max(0, telomere_ratio - (error_penalty * 0.5))

        return TelomereStatus(
            phase=self._phase,
            telomere_length=self._telomere_length,
            max_telomere_length=self.max_operations,
            operations_remaining=self._telomere_length,
            time_remaining=time_remaining,
            health_score=health,
            senescence_reason=self._senescence_reason
        )

    def get_phase(self) -> LifecyclePhase:
        """Get current lifecycle phase."""
        return self._phase

    def is_active(self) -> bool:
        """Check if agent is in active phase."""
        return self._phase == LifecyclePhase.ACTIVE

    def is_operational(self) -> bool:
        """Check if agent can still perform operations."""
        return self._phase in (LifecyclePhase.NASCENT, LifecyclePhase.ACTIVE, LifecyclePhase.SENESCENT)

    def get_age(self) -> timedelta | None:
        """Get the agent's age since start."""
        if not self._started_at:
            return None
        return datetime.now() - self._started_at

    def get_statistics(self) -> dict:
        """Get telomere statistics."""
        return {
            "phase": self._phase.value,
            "telomere_length": self._telomere_length,
            "max_telomere_length": self.max_operations,
            "utilization": 1 - (self._telomere_length / max(1, self.max_operations)),
            "operations_count": self._operations_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._operations_count),
            "renewal_count": self._renewal_count,
            "age_seconds": self.get_age().total_seconds() if self.get_age() else 0,
            "senescence_reason": self._senescence_reason.value if self._senescence_reason else None,
            "events_count": len(self._events),
        }

    def get_events(self, limit: int = 100) -> list[LifecycleEvent]:
        """Get recent lifecycle events."""
        return self._events[-limit:]

    def reset(self):
        """Reset telomere to initial state (for testing)."""
        with self._lock:
            self._telomere_length = self.max_operations
            self._phase = LifecyclePhase.NASCENT
            self._senescence_reason = None
            self._started_at = None
            self._last_activity = None
            self._terminated_at = None
            self._operations_count = 0
            self._error_count = 0
            self._events.clear()
            self._log_event("reset")
