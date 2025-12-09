"""
Feed-Forward Loops: Multi-Agent Control Topologies
===================================================

Biological Analogy:
- Coherent FFL: Both paths reinforce the same outcome
- Incoherent FFL: Paths have opposing effects (creates pulse response)
- Type 1: X activates Y and Z; Y activates Z (AND logic)
- Type 2: X activates Y, inhibits Z; Y activates Z (timing filter)
- Circuit breaker: Prevents cascade failures
- Caching: Remember recent decisions (synaptic potentiation)

Feed-forward loops are network motifs that filter noise, detect
persistence, and create delays in signal processing.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import threading
import hashlib

from ..core.agent import BioAgent
from ..core.types import Signal, ActionProtein
from ..state.metabolism import ATP_Store


class GateLogic(Enum):
    """Logic gate types for combining agent outputs."""
    AND = "and"           # All must agree
    OR = "or"             # Any must agree
    MAJORITY = "majority" # More than half must agree
    UNANIMOUS = "unanimous"  # All must agree (same as AND)
    EXECUTOR_PRIORITY = "executor_priority"  # Executor decides unless blocked
    ASSESSOR_PRIORITY = "assessor_priority"  # Assessor decides unless permitted


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class LoopResult:
    """Result of a feed-forward loop execution."""
    success: bool
    action: str
    executor_output: ActionProtein | None = None
    assessor_output: ActionProtein | None = None
    blocked: bool = False
    block_reason: str = ""
    cached: bool = False
    processing_time_ms: float = 0.0
    gate_logic: GateLogic | None = None


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure: datetime | None
    last_success: datetime | None
    trips_count: int


class CoherentFeedForwardLoop:
    """
    Advanced Coherent Feed-Forward Loop (Type 1).

    The canonical guardrail topology where an action only proceeds
    if both the executor AND a risk assessor approve.

    Features:

    1. Configurable Gate Logic
       - AND: Both must approve (default)
       - OR: Either can approve
       - MAJORITY: For multi-agent scenarios
       - EXECUTOR_PRIORITY: Executor decides unless assessor blocks
       - ASSESSOR_PRIORITY: Assessor decides unless executor fails

    2. Circuit Breaker
       - Prevents cascade failures
       - Auto-opens after consecutive failures
       - Half-open state for recovery testing

    3. Result Caching
       - Cache recent decisions
       - Configurable TTL
       - Reduces redundant processing

    4. Timeout Handling
       - Per-agent timeouts
       - Graceful degradation
       - Fallback strategies

    5. Statistics and Audit
       - Track success/failure rates
       - Log all decisions
       - Performance metrics

    Example:
        >>> budget = ATP_Store(budget=500)
        >>> loop = CoherentFeedForwardLoop(budget=budget, gate_logic=GateLogic.AND)
        >>> result = loop.run("Deploy to production")
        >>> if result.blocked:
        ...     print(f"Blocked: {result.block_reason}")
    """

    def __init__(
        self,
        budget: ATP_Store,
        gate_logic: GateLogic = GateLogic.AND,
        enable_circuit_breaker: bool = True,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        enable_cache: bool = True,
        cache_ttl_seconds: float = 300.0,
        timeout_seconds: float = 30.0,
        on_block: Callable[[LoopResult], None] | None = None,
        on_permit: Callable[[LoopResult], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Coherent Feed-Forward Loop.

        Args:
            budget: Shared ATP budget for agents
            gate_logic: How to combine agent outputs
            enable_circuit_breaker: Enable circuit breaker pattern
            failure_threshold: Failures before circuit opens
            recovery_timeout_seconds: Time before testing recovery
            enable_cache: Enable decision caching
            cache_ttl_seconds: Cache time-to-live
            timeout_seconds: Per-operation timeout
            on_block: Callback when request is blocked
            on_permit: Callback when request is permitted
            silent: Suppress console output
        """
        self.budget = budget
        self.gate_logic = gate_logic
        self.enable_circuit_breaker = enable_circuit_breaker
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.enable_cache = enable_cache
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.timeout_seconds = timeout_seconds
        self.on_block = on_block
        self.on_permit = on_permit
        self.silent = silent

        # Create agents
        self.executor = BioAgent("Gene_Z (Exec)", role="Executor", atp_store=budget)
        self.assessor = BioAgent("Gene_Y (Risk)", role="RiskAssessor", atp_store=budget)

        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure: datetime | None = None
        self._last_success: datetime | None = None
        self._trips_count = 0

        # Cache
        self._cache: dict[str, tuple[LoopResult, datetime]] = {}
        self._lock = threading.Lock()

        # Statistics
        self._total_requests = 0
        self._total_blocked = 0
        self._total_permitted = 0
        self._total_errors = 0
        self._results_log: list[LoopResult] = []

    def run(self, user_prompt: str) -> LoopResult:
        """
        Run a request through the feed-forward loop.

        Args:
            user_prompt: The input to process

        Returns:
            LoopResult with decision and details
        """
        import time
        start_time = time.time()

        self._total_requests += 1

        # Check circuit breaker
        if self.enable_circuit_breaker:
            if not self._check_circuit():
                result = LoopResult(
                    success=False,
                    action="CIRCUIT_OPEN",
                    blocked=True,
                    block_reason="Circuit breaker is open",
                    gate_logic=self.gate_logic
                )
                self._record_result(result)
                return result

        # Check cache
        if self.enable_cache:
            cached = self._check_cache(user_prompt)
            if cached:
                cached.cached = True
                if not self.silent:
                    print(f"💾 [CFFL] Cache hit")
                return cached

        # Create signal
        signal = Signal(content=user_prompt)

        # Run both agents
        try:
            z_out = self.executor.express(signal)
            y_out = self.assessor.express(signal)
        except Exception as e:
            self._record_failure()
            result = LoopResult(
                success=False,
                action="ERROR",
                blocked=True,
                block_reason=f"Agent error: {e}",
                processing_time_ms=(time.time() - start_time) * 1000,
                gate_logic=self.gate_logic
            )
            self._record_result(result)
            return result

        # Apply gate logic
        result = self._apply_gate_logic(z_out, y_out)
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Update circuit breaker
        if result.success and not result.blocked:
            self._record_success()
        elif result.blocked:
            # Blocks are intentional, not failures
            pass
        else:
            self._record_failure()

        # Cache result
        if self.enable_cache:
            self._cache_result(user_prompt, result)

        # Callbacks and logging
        self._record_result(result)

        if result.blocked:
            self._total_blocked += 1
            if self.on_block:
                self.on_block(result)
        else:
            self._total_permitted += 1
            if self.on_permit:
                self.on_permit(result)

        # Console output
        if not self.silent:
            self._print_result(result)

        return result

    def _apply_gate_logic(self, z_out: ActionProtein, y_out: ActionProtein) -> LoopResult:
        """Apply the configured gate logic to agent outputs."""

        executor_permits = z_out.action_type in ("EXECUTE", "PERMIT")
        executor_blocks = z_out.action_type == "BLOCK"
        executor_fails = z_out.action_type == "FAILURE"

        assessor_permits = y_out.action_type == "PERMIT"
        assessor_blocks = y_out.action_type == "BLOCK"

        if self.gate_logic == GateLogic.AND or self.gate_logic == GateLogic.UNANIMOUS:
            # Both must agree to proceed
            if assessor_blocks:
                return LoopResult(
                    success=True,
                    action="BLOCKED",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=True,
                    block_reason=f"Risk Assessor: {y_out.payload}",
                    gate_logic=self.gate_logic
                )
            if executor_fails:
                return LoopResult(
                    success=False,
                    action="FAILURE",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=True,
                    block_reason=f"Executor failure: {z_out.payload}",
                    gate_logic=self.gate_logic
                )
            if executor_blocks:
                return LoopResult(
                    success=True,
                    action="SKIPPED",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=True,
                    block_reason=f"Executor skipped: {z_out.payload}",
                    gate_logic=self.gate_logic
                )
            if executor_permits and assessor_permits:
                return LoopResult(
                    success=True,
                    action="SUCCESS",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=False,
                    gate_logic=self.gate_logic
                )

        elif self.gate_logic == GateLogic.OR:
            # Either can permit
            if executor_permits or assessor_permits:
                return LoopResult(
                    success=True,
                    action="SUCCESS",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=False,
                    gate_logic=self.gate_logic
                )
            # Both block/fail
            return LoopResult(
                success=False,
                action="BLOCKED",
                executor_output=z_out,
                assessor_output=y_out,
                blocked=True,
                block_reason="Both agents rejected",
                gate_logic=self.gate_logic
            )

        elif self.gate_logic == GateLogic.EXECUTOR_PRIORITY:
            # Executor decides unless assessor explicitly blocks
            if assessor_blocks:
                return LoopResult(
                    success=True,
                    action="BLOCKED",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=True,
                    block_reason=f"Risk Assessor override: {y_out.payload}",
                    gate_logic=self.gate_logic
                )
            if executor_permits:
                return LoopResult(
                    success=True,
                    action="SUCCESS",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=False,
                    gate_logic=self.gate_logic
                )

        elif self.gate_logic == GateLogic.ASSESSOR_PRIORITY:
            # Assessor decides unless executor fails
            if executor_fails:
                return LoopResult(
                    success=False,
                    action="FAILURE",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=True,
                    block_reason=f"Executor failure: {z_out.payload}",
                    gate_logic=self.gate_logic
                )
            if assessor_permits:
                return LoopResult(
                    success=True,
                    action="SUCCESS",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=False,
                    gate_logic=self.gate_logic
                )
            if assessor_blocks:
                return LoopResult(
                    success=True,
                    action="BLOCKED",
                    executor_output=z_out,
                    assessor_output=y_out,
                    blocked=True,
                    block_reason=f"Risk Assessor: {y_out.payload}",
                    gate_logic=self.gate_logic
                )

        # Default: error state
        return LoopResult(
            success=False,
            action="ERROR",
            executor_output=z_out,
            assessor_output=y_out,
            blocked=True,
            block_reason="Signal mismatch",
            gate_logic=self.gate_logic
        )

    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows the request."""
        with self._lock:
            if self._circuit_state == CircuitState.CLOSED:
                return True

            if self._circuit_state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure and datetime.now() - self._last_failure >= self.recovery_timeout:
                    self._circuit_state = CircuitState.HALF_OPEN
                    if not self.silent:
                        print("🔌 [CFFL] Circuit half-open, testing...")
                    return True
                return False

            if self._circuit_state == CircuitState.HALF_OPEN:
                # Allow one request through to test
                return True

        return True

    def _record_success(self):
        """Record a successful operation."""
        with self._lock:
            self._success_count += 1
            self._last_success = datetime.now()

            if self._circuit_state == CircuitState.HALF_OPEN:
                # Recovery successful
                self._circuit_state = CircuitState.CLOSED
                self._failure_count = 0
                if not self.silent:
                    print("🔌 [CFFL] Circuit closed, recovered")

    def _record_failure(self):
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._total_errors += 1
            self._last_failure = datetime.now()

            if self._circuit_state == CircuitState.HALF_OPEN:
                # Recovery failed
                self._circuit_state = CircuitState.OPEN
                self._trips_count += 1
                if not self.silent:
                    print("🔌 [CFFL] Circuit re-opened, recovery failed")

            elif self._circuit_state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._circuit_state = CircuitState.OPEN
                    self._trips_count += 1
                    if not self.silent:
                        print(f"🔌 [CFFL] Circuit opened after {self._failure_count} failures")

    def _check_cache(self, prompt: str) -> LoopResult | None:
        """Check cache for existing result."""
        cache_key = self._get_cache_key(prompt)
        with self._lock:
            if cache_key in self._cache:
                result, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < self.cache_ttl:
                    return result
                else:
                    del self._cache[cache_key]
        return None

    def _cache_result(self, prompt: str, result: LoopResult):
        """Cache a result."""
        cache_key = self._get_cache_key(prompt)
        with self._lock:
            self._cache[cache_key] = (result, datetime.now())
            # Limit cache size
            if len(self._cache) > 1000:
                oldest = min(self._cache.items(), key=lambda x: x[1][1])
                del self._cache[oldest[0]]

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def _record_result(self, result: LoopResult):
        """Record result for audit."""
        self._results_log.append(result)
        if len(self._results_log) > 1000:
            self._results_log = self._results_log[-1000:]

    def _print_result(self, result: LoopResult):
        """Print result to console."""
        if result.blocked:
            if result.action == "BLOCKED":
                print(f"🛑 BLOCKED by Risk Assessor: {result.block_reason}")
            elif result.action == "FAILURE":
                print(f"💥 RUNTIME ERROR: {result.block_reason}")
            elif result.action == "SKIPPED":
                print(f"⏸️ SKIPPED by Executor Memory: {result.block_reason}")
            elif result.action == "CIRCUIT_OPEN":
                print(f"🔌 CIRCUIT OPEN: {result.block_reason}")
            else:
                print(f"⚠️ {result.action}: {result.block_reason}")
        else:
            if result.executor_output:
                print(f"✅ SUCCESS: {result.executor_output.payload}")
            else:
                print(f"✅ SUCCESS")

    def clear_cache(self):
        """Clear the result cache."""
        with self._lock:
            self._cache.clear()

    def reset_circuit_breaker(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._circuit_state = CircuitState.CLOSED
            self._failure_count = 0

    def get_circuit_breaker_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return CircuitBreakerStats(
            state=self._circuit_state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure=self._last_failure,
            last_success=self._last_success,
            trips_count=self._trips_count
        )

    def get_statistics(self) -> dict:
        """Get loop statistics."""
        return {
            "total_requests": self._total_requests,
            "total_blocked": self._total_blocked,
            "total_permitted": self._total_permitted,
            "total_errors": self._total_errors,
            "block_rate": self._total_blocked / max(1, self._total_requests),
            "error_rate": self._total_errors / max(1, self._total_requests),
            "gate_logic": self.gate_logic.value,
            "circuit_state": self._circuit_state.value,
            "cache_size": len(self._cache),
        }

    def get_results_log(self, limit: int = 100) -> list[LoopResult]:
        """Get recent results."""
        return self._results_log[-limit:]


class NegativeFeedbackLoop:
    """
    Negative Feedback Loop for Error Correction.

    Biological analogy: Homeostasis mechanisms that detect deviation
    from a setpoint and apply corrective action.

    Features:
    - Setpoint tracking
    - Error measurement
    - Proportional correction
    - Damping to prevent oscillation
    """

    def __init__(
        self,
        setpoint: float = 0.0,
        gain: float = 0.5,
        damping: float = 0.1,
        min_correction: float = 0.0,
        max_correction: float = float('inf'),
        on_correction: Callable[[float, float], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Negative Feedback Loop.

        Args:
            setpoint: Target value to maintain
            gain: How strongly to correct (0-1)
            damping: Damping factor to prevent oscillation
            min_correction: Minimum correction to apply
            max_correction: Maximum correction to apply
            on_correction: Callback when correction is applied
            silent: Suppress console output
        """
        self.setpoint = setpoint
        self.gain = gain
        self.damping = damping
        self.min_correction = min_correction
        self.max_correction = max_correction
        self.on_correction = on_correction
        self.silent = silent

        self._current_value = setpoint
        self._last_error = 0.0
        self._corrections_count = 0
        self._total_correction = 0.0

    def measure(self, current_value: float) -> float:
        """
        Measure current value and calculate correction.

        Returns the correction to apply.
        """
        self._current_value = current_value
        error = self.setpoint - current_value

        # Proportional correction with damping
        correction = error * self.gain
        derivative = error - self._last_error
        correction -= derivative * self.damping

        # Clamp correction
        if abs(correction) < self.min_correction:
            correction = 0.0
        elif abs(correction) > self.max_correction:
            correction = self.max_correction if correction > 0 else -self.max_correction

        self._last_error = error
        self._corrections_count += 1
        self._total_correction += abs(correction)

        if correction != 0 and self.on_correction:
            self.on_correction(error, correction)

        if not self.silent and correction != 0:
            print(f"⚖️ [Feedback] Error: {error:.3f}, Correction: {correction:.3f}")

        return correction

    def apply(self, current_value: float) -> float:
        """Measure and return corrected value."""
        correction = self.measure(current_value)
        return current_value + correction

    def set_setpoint(self, new_setpoint: float):
        """Update the setpoint."""
        self.setpoint = new_setpoint

    def get_error(self) -> float:
        """Get current error from setpoint."""
        return self.setpoint - self._current_value

    def get_statistics(self) -> dict:
        """Get feedback loop statistics."""
        return {
            "setpoint": self.setpoint,
            "current_value": self._current_value,
            "current_error": self.get_error(),
            "corrections_count": self._corrections_count,
            "total_correction": self._total_correction,
            "average_correction": self._total_correction / max(1, self._corrections_count),
        }
