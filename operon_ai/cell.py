"""Integrated Cell - combines Quality, Surveillance, and Coordination systems."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from operon_ai.quality import UbiquitinPool, UbiquitinTag, TaggedData, DegradationResult
from operon_ai.quality.components import ProvenanceContext
from operon_ai.quality.proteasome import Proteasome

from operon_ai.surveillance import ImmuneSystem, ThreatLevel, ResponseAction
from operon_ai.surveillance.tcell import ImmuneResponse

from operon_ai.coordination import CoordinationSystem, CoordinationResult
from operon_ai.coordination.watchdog import ApoptosisEvent


@dataclass
class CellExecutionResult:
    """Result of cell-coordinated execution."""

    agent_id: str
    success: bool
    output: Any = None
    tagged_output: Optional[TaggedData] = None

    # Subsystem results
    degradation_result: Optional[DegradationResult] = None
    coordination_result: Optional[CoordinationResult] = None
    immune_response: Optional[ImmuneResponse] = None

    # Blocking info
    blocked_by: Optional[str] = None  # "surveillance", "coordination", "quality"
    error: Optional[str] = None

    execution_time_ms: Optional[float] = None


@dataclass
class CellHealth:
    """Health status of the integrated cell."""

    healthy: bool
    surveillance_alerts: list[ImmuneResponse] = field(default_factory=list)
    apoptosis_events: list[ApoptosisEvent] = field(default_factory=list)
    pool_status: dict = field(default_factory=dict)
    coordination_stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "HEALTHY" if self.healthy else "UNHEALTHY"
        return f"CellHealth({status}, alerts={len(self.surveillance_alerts)})"


@dataclass
class IntegratedCell:
    """
    Integrated cell combining Quality, Surveillance, and Coordination.

    This is the main organelle that orchestrates all three systems
    for robust agent operation management.
    """

    # Configuration
    pool_capacity: int = 1000
    degradation_threshold: float = 0.3
    max_operation_time: Optional[timedelta] = None

    # Subsystems
    quality_pool: UbiquitinPool = field(init=False)
    proteasome: Proteasome = field(init=False)
    surveillance: ImmuneSystem = field(init=False)
    coordination: CoordinationSystem = field(init=False)

    # Cross-system state
    agent_operations: dict[str, str] = field(default_factory=dict)  # agent_id -> operation_id

    def __post_init__(self):
        # Initialize subsystems
        self.quality_pool = UbiquitinPool(capacity=self.pool_capacity)
        self.proteasome = Proteasome(degradation_threshold=self.degradation_threshold)
        self.surveillance = ImmuneSystem()
        self.coordination = CoordinationSystem(
            max_operation_time=self.max_operation_time,
        )

    def register_agent(self, agent_id: str) -> None:
        """Register an agent with all subsystems."""
        self.surveillance.register_agent(agent_id)

    def register_resource(self, resource_id: str, allow_preemption: bool = False) -> None:
        """Register a resource for coordination."""
        self.coordination.register_resource(resource_id, allow_preemption)

    def execute(
        self,
        agent_id: str,
        operation_id: str,
        work_fn: Callable[[], Any],
        resources: Optional[list[str]] = None,
        validate_fn: Optional[Callable[[Any], bool]] = None,
        priority: int = 0,
    ) -> CellExecutionResult:
        """
        Execute an operation with full cell integration.

        1. Pre-execution: Check surveillance health
        2. Coordination: Manage phase transitions and resources
        3. Quality: Tag output with provenance
        4. Post-execution: Record observation, check quality
        """
        import time
        start_time = time.time()

        # Track agent operation
        self.agent_operations[agent_id] = operation_id

        try:
            # Pre-execution: Check if agent is under surveillance alert
            # (In full implementation, would check threat level)

            # Execute through coordination system
            coord_result = self.coordination.execute_operation(
                operation_id=operation_id,
                agent_id=agent_id,
                work_fn=work_fn,
                resources=resources,
                validate_fn=validate_fn,
                priority=priority,
            )

            if not coord_result.success:
                return CellExecutionResult(
                    agent_id=agent_id,
                    success=False,
                    blocked_by="coordination",
                    error=coord_result.error,
                    coordination_result=coord_result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            output = coord_result.result

            # Tag output with provenance
            tag = self.quality_pool.allocate(
                origin=agent_id,
                confidence=1.0,
            )

            tagged_output = None
            if tag:
                tagged_output = TaggedData(data=output, tag=tag)

            # Record observation in surveillance
            if agent_id in self.surveillance.displays:
                output_str = str(output) if output else ""
                response_time = (time.time() - start_time)
                self.surveillance.record_observation(
                    agent_id=agent_id,
                    output=output_str,
                    response_time=response_time,
                    confidence=tag.confidence if tag else 0.5,
                )

            # Quality check through proteasome (if tag exists)
            degradation_result = None
            if tag and tagged_output:
                ctx = ProvenanceContext(
                    tag=tag,
                    source_module=agent_id,
                    target_module="cell_output",
                )
                _, _, degradation_result = self.proteasome.inspect(
                    output, tag, ctx, self.quality_pool
                )

            execution_time_ms = (time.time() - start_time) * 1000

            return CellExecutionResult(
                agent_id=agent_id,
                success=True,
                output=output,
                tagged_output=tagged_output,
                degradation_result=degradation_result,
                coordination_result=coord_result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            return CellExecutionResult(
                agent_id=agent_id,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        finally:
            # Clean up agent operation tracking
            if agent_id in self.agent_operations:
                del self.agent_operations[agent_id]

    def run_maintenance(self) -> dict:
        """
        Run periodic maintenance across all subsystems.

        - Coordination: Priority inheritance, watchdog
        - Surveillance: (Future: memory pruning)
        - Quality: (Future: pool rebalancing)
        """
        events = {}

        # Coordination maintenance
        coord_events = self.coordination.run_maintenance()
        events["coordination"] = coord_events

        # Handle cross-system effects
        for apoptosis_event in coord_events.get("apoptosis", []):
            # Agent terminated - could affect surveillance
            self._handle_agent_termination(apoptosis_event)

        return events

    def _handle_agent_termination(self, event: ApoptosisEvent) -> None:
        """Handle cross-system effects of agent termination."""
        # Could record in surveillance, recycle quality tags, etc.
        pass

    def health(self) -> CellHealth:
        """Get integrated health status."""
        # Check each subsystem
        pool_status = self.quality_pool.status()
        coord_stats = self.coordination.health()
        surveillance_health = self.surveillance.health()

        # Determine overall health
        healthy = (
            pool_status["available"] > 0 and
            coord_stats["active_operations"] < 1000  # Reasonable limit
        )

        return CellHealth(
            healthy=healthy,
            pool_status=pool_status,
            coordination_stats=coord_stats,
        )

    def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        self.coordination.shutdown()
