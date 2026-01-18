"""
Regenerative Swarm: Metabolic Self-Healing through Apoptosis and Regeneration
=============================================================================

Biological Analogy:
- Apoptosis: Programmed cell death - a clean, controlled process that removes
  damaged or dysfunctional cells without causing inflammation.
- Stem Cell Regeneration: After apoptosis, stem cells divide to replace the
  lost tissue, restoring function.
- Cellular Debris Signaling: Dying cells release signals that inform neighboring
  cells about the threat, enabling adaptive response.

The Regenerative Swarm extends basic timeout-based termination into a full
regeneration cycle:
1. Detect stuck workers via entropy monitoring (repeated outputs = no progress)
2. Trigger clean apoptosis (preserve useful state, discard corrupted state)
3. Summarize failed worker's memory
4. Spawn fresh worker with injected summary ("Worker_1 died trying X. Try Y.")

The key insight: Agent death is not just cleanup - it's information transfer.
The dying agent's experience becomes a lesson for its successor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol
from enum import Enum
import hashlib


class WorkerStatus(Enum):
    """Status of a worker in the swarm."""

    ACTIVE = "active"
    STUCK = "stuck"  # Detected as not making progress
    APOPTOTIC = "apoptotic"  # In process of clean shutdown
    TERMINATED = "terminated"  # Fully shut down


class ApoptosisReason(Enum):
    """Reason for worker termination."""

    ENTROPY_COLLAPSE = "entropy_collapse"  # Output becoming repetitive
    TIMEOUT = "timeout"  # Exceeded time limit
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # ATP depleted
    ERROR_ACCUMULATION = "error_accumulation"  # Too many errors
    MANUAL = "manual"


@dataclass
class WorkerMemory:
    """Memory state of a worker, extractable for regeneration."""

    task_history: list[str] = field(default_factory=list)
    output_history: list[str] = field(default_factory=list)
    error_history: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)

    def add_attempt(self, task: str, output: str, error: str | None = None):
        """Record an attempt."""
        self.task_history.append(task)
        self.output_history.append(output)
        if error:
            self.error_history.append(error)


@dataclass
class ApoptosisEvent:
    """Record of a worker termination."""

    worker_id: str
    reason: ApoptosisReason
    memory_summary: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


class Worker(Protocol):
    """Protocol for workers in the swarm."""

    @property
    def id(self) -> str:
        """Unique identifier for this worker."""
        ...

    @property
    def memory(self) -> WorkerMemory:
        """Access worker's memory state."""
        ...

    def step(self, task: str) -> str:
        """
        Execute one step of work.

        Args:
            task: The task to work on

        Returns:
            Output/result of this step
        """
        ...


@dataclass
class SimpleWorker:
    """
    Simple worker implementation for demonstration.

    A real implementation would wrap an LLM agent.
    """

    id: str
    work_function: Callable[[str, WorkerMemory], str]
    memory: WorkerMemory = field(default_factory=WorkerMemory)
    status: WorkerStatus = WorkerStatus.ACTIVE

    def step(self, task: str) -> str:
        """Execute one step of work."""
        output = self.work_function(task, self.memory)
        self.memory.add_attempt(task, output)
        return output


@dataclass
class RegenerationEvent:
    """Record of a worker regeneration."""

    old_worker_id: str
    new_worker_id: str
    injected_summary: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SwarmResult:
    """Result of swarm execution."""

    success: bool
    output: str | None
    total_workers_spawned: int
    apoptosis_events: list[ApoptosisEvent]
    regeneration_events: list[RegenerationEvent]
    final_worker_id: str | None


@dataclass
class RegenerativeSwarm:
    """
    Supervisor that detects stuck workers and regenerates them.

    Biological parallel: Apoptosis (programmed cell death) + stem cell
    regeneration to replace damaged cells. The dying cell's state is not
    entirely lost - cellular debris signals neighboring cells about the threat.

    Example:
        >>> def create_worker(name: str, memory_hints: list[str]) -> SimpleWorker:
        ...     def work(task: str, memory: WorkerMemory) -> str:
        ...         if "try different" in str(memory_hints):
        ...             return "SUCCESS: Alternative approach worked"
        ...         return "STUCK: Still thinking..."  # Will trigger apoptosis
        ...     return SimpleWorker(id=name, work_function=work)
        ...
        >>> swarm = RegenerativeSwarm(
        ...     worker_factory=create_worker,
        ...     summarizer=lambda mem: [f"Previous worker tried: {mem.task_history}"],
        ...     entropy_threshold=0.9,  # High similarity = stuck
        ...     max_steps_per_worker=5,
        ... )
        >>> result = swarm.supervise("Solve the puzzle")
    """

    worker_factory: Callable[[str, list[str]], Worker]
    summarizer: Callable[[WorkerMemory], list[str]]
    entropy_threshold: float = 0.9  # Similarity threshold triggering apoptosis
    max_steps_per_worker: int = 10
    max_regenerations: int = 3
    step_timeout: timedelta | None = None
    silent: bool = False

    # Internal state
    _worker_counter: int = field(default=0, init=False)
    _apoptosis_events: list[ApoptosisEvent] = field(default_factory=list, init=False)
    _regeneration_events: list[RegenerationEvent] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._worker_counter = 0
        self._apoptosis_events = []
        self._regeneration_events = []

    def supervise(self, task: str) -> SwarmResult:
        """
        Run workers on task with automatic regeneration.

        1. Spawn initial worker
        2. Run steps, monitoring entropy
        3. If stuck detected, trigger apoptosis and regenerate
        4. Continue until success or max regenerations

        Args:
            task: The task to complete

        Returns:
            SwarmResult with outcome and history
        """
        memory_hints: list[str] = []
        regenerations = 0

        while regenerations <= self.max_regenerations:
            # Spawn worker
            worker = self._spawn_worker(memory_hints)

            if not self.silent:
                print(
                    f"🧬 [RegenerativeSwarm] Spawned worker {worker.id}"
                    + (f" with hints: {memory_hints[:2]}..." if memory_hints else "")
                )

            # Run worker
            result = self._run_worker(worker, task)

            if result is not None:
                # Success!
                return SwarmResult(
                    success=True,
                    output=result,
                    total_workers_spawned=self._worker_counter,
                    apoptosis_events=self._apoptosis_events,
                    regeneration_events=self._regeneration_events,
                    final_worker_id=worker.id,
                )

            # Worker failed - trigger apoptosis
            apoptosis_event = self._trigger_apoptosis(
                worker, ApoptosisReason.ENTROPY_COLLAPSE
            )
            self._apoptosis_events.append(apoptosis_event)

            # Summarize memory for next worker
            memory_hints = apoptosis_event.memory_summary

            # Record regeneration
            old_id = worker.id
            regenerations += 1

            if regenerations <= self.max_regenerations:
                self._regeneration_events.append(
                    RegenerationEvent(
                        old_worker_id=old_id,
                        new_worker_id=f"worker_{self._worker_counter + 1}",
                        injected_summary=memory_hints,
                    )
                )

        # All regenerations exhausted
        if not self.silent:
            print(
                f"🧬 [RegenerativeSwarm] Max regenerations ({self.max_regenerations}) "
                f"exhausted. Task failed."
            )

        return SwarmResult(
            success=False,
            output=None,
            total_workers_spawned=self._worker_counter,
            apoptosis_events=self._apoptosis_events,
            regeneration_events=self._regeneration_events,
            final_worker_id=None,
        )

    def _spawn_worker(self, memory_hints: list[str]) -> Worker:
        """Spawn a new worker with optional memory hints."""
        self._worker_counter += 1
        name = f"worker_{self._worker_counter}"
        return self.worker_factory(name, memory_hints)

    def _run_worker(self, worker: Worker, task: str) -> str | None:
        """
        Run worker until completion, stuck detection, or step limit.

        Returns output on success, None if stuck/failed.
        """
        recent_outputs: list[str] = []

        for _ in range(self.max_steps_per_worker):
            output = worker.step(task)

            # Check for explicit success markers
            if self._is_success(output):
                return output

            # Track recent outputs for entropy calculation
            recent_outputs.append(output)
            if len(recent_outputs) > 3:
                recent_outputs.pop(0)

            # Check for entropy collapse (stuck)
            if len(recent_outputs) >= 3:
                entropy = self._calculate_entropy(recent_outputs)
                if entropy < (1 - self.entropy_threshold):
                    if not self.silent:
                        print(
                            f"🧬 [RegenerativeSwarm] Entropy collapse detected for {worker.id} "
                            f"(entropy: {entropy:.2f})"
                        )
                    return None

        # Step limit reached without success
        if not self.silent:
            print(
                f"🧬 [RegenerativeSwarm] Step limit ({self.max_steps_per_worker}) "
                f"reached for {worker.id}"
            )
        return None

    def _is_success(self, output: str) -> bool:
        """Check if output indicates task completion."""
        # Simple heuristic - real implementation would use task-specific logic
        success_markers = ["SUCCESS", "SOLVED", "COMPLETE", "DONE", "FINISHED"]
        return any(marker in output.upper() for marker in success_markers)

    def _calculate_entropy(self, outputs: list[str]) -> float:
        """
        Calculate output entropy (diversity).

        Low entropy = outputs are similar = agent is stuck.
        High entropy = outputs are diverse = agent is exploring.

        Returns value between 0 (identical) and 1 (completely different).
        """
        if len(outputs) < 2:
            return 1.0

        # Use hash-based similarity
        hashes = [hashlib.md5(o.encode()).hexdigest()[:8] for o in outputs]
        unique_hashes = len(set(hashes))

        return unique_hashes / len(hashes)

    def _trigger_apoptosis(
        self, worker: Worker, reason: ApoptosisReason
    ) -> ApoptosisEvent:
        """
        Trigger clean shutdown of a worker.

        Extracts useful information before termination.
        """
        if not self.silent:
            print(
                f"🧬 [RegenerativeSwarm] APOPTOSIS - Terminating {worker.id} "
                f"(reason: {reason.value})"
            )

        # Summarize memory for next generation
        memory_summary = self.summarizer(worker.memory)

        return ApoptosisEvent(
            worker_id=worker.id,
            reason=reason,
            memory_summary=memory_summary,
            details=f"Terminated after {len(worker.memory.task_history)} steps",
        )


def create_default_summarizer() -> Callable[[WorkerMemory], list[str]]:
    """
    Create a default memory summarizer.

    In a real implementation, this would use an LLM to intelligently
    summarize the failed worker's experience.
    """

    def summarizer(memory: WorkerMemory) -> list[str]:
        hints = []

        # Summarize what was tried
        if memory.task_history:
            hints.append(f"Previous worker attempted: {len(memory.task_history)} steps")

        # Summarize errors
        if memory.error_history:
            hints.append(f"Encountered errors: {memory.error_history[-1]}")

        # Suggest trying different approach
        if memory.output_history:
            last_outputs = memory.output_history[-3:]
            if len(set(last_outputs)) == 1:
                hints.append(
                    "Worker got stuck repeating same output. Try a different approach."
                )

        return hints

    return summarizer
