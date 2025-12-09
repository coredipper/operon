"""
Cascade: Signal Amplification and Multi-Stage Processing
=========================================================

Biological Analogy:
- MAPK cascade: Sequential kinase activation amplifying signals
- Phosphorylation cascade: Each step modifies and activates the next
- Complement cascade: Immune amplification system
- Coagulation cascade: Blood clotting through sequential activation
- Signal amplification: Weak input → strong output

The Cascade topology enables multi-stage signal processing with
amplification, transformation, and checkpoint gates at each stage.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Generic, TypeVar
from enum import Enum
from datetime import datetime
import threading
import time

from ..core.agent import BioAgent
from ..core.types import Signal, ActionProtein
from ..state.metabolism import ATP_Store


T = TypeVar('T')


class StageStatus(Enum):
    """Status of a cascade stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class CascadeMode(Enum):
    """How the cascade processes signals."""
    SEQUENTIAL = "sequential"     # Each stage waits for previous
    PARALLEL = "parallel"         # All stages run simultaneously
    CONDITIONAL = "conditional"   # Stages run based on conditions
    AMPLIFYING = "amplifying"     # Each stage amplifies the signal


@dataclass
class StageResult:
    """Result from a single cascade stage."""
    stage_name: str
    status: StageStatus
    input_signal: Any
    output_signal: Any
    amplification_factor: float = 1.0
    processing_time_ms: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class CascadeResult:
    """Result from complete cascade execution."""
    success: bool
    final_output: Any
    stages_completed: int
    stages_total: int
    total_amplification: float
    total_time_ms: float
    stage_results: list[StageResult] = field(default_factory=list)
    blocked_at: str | None = None


@dataclass
class CascadeStage:
    """Definition of a cascade stage."""
    name: str
    processor: Callable[[Any], Any]
    amplification: float = 1.0
    checkpoint: Callable[[Any], bool] | None = None  # Gate function
    on_error: Callable[[Exception], Any] | None = None  # Error handler
    timeout_seconds: float = 30.0
    required: bool = True  # If False, can skip on failure


class Cascade:
    """
    Multi-Stage Signal Processing Cascade.

    Implements sequential signal amplification and transformation
    through multiple processing stages, similar to biological
    signal cascades.

    Features:

    1. Multi-Stage Processing
       - Define sequential processing stages
       - Each stage transforms the signal
       - Configurable stage order

    2. Signal Amplification
       - Each stage can amplify the signal
       - Cumulative amplification tracking
       - Automatic gain control

    3. Checkpoint Gates
       - Optional gates between stages
       - Block cascade if conditions not met
       - Custom gate logic

    4. Error Handling
       - Per-stage error handlers
       - Cascade halt on critical errors
       - Recovery and retry mechanisms

    5. Parallel Processing
       - Sequential or parallel modes
       - Fork and join patterns
       - Conditional branching

    Example:
        >>> cascade = Cascade("SignalProcessor")
        >>> cascade.add_stage(CascadeStage(
        ...     name="normalize",
        ...     processor=lambda x: x.lower(),
        ...     amplification=1.0
        ... ))
        >>> cascade.add_stage(CascadeStage(
        ...     name="validate",
        ...     processor=lambda x: x if len(x) > 0 else None,
        ...     checkpoint=lambda x: x is not None
        ... ))
        >>> result = cascade.run("HELLO WORLD")
        >>> print(result.final_output)
        'hello world'
    """

    def __init__(
        self,
        name: str,
        mode: CascadeMode = CascadeMode.SEQUENTIAL,
        max_amplification: float = 100.0,
        halt_on_failure: bool = True,
        on_stage_complete: Callable[[StageResult], None] | None = None,
        on_cascade_complete: Callable[[CascadeResult], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Cascade.

        Args:
            name: Name of this cascade
            mode: Processing mode (sequential, parallel, etc.)
            max_amplification: Maximum cumulative amplification
            halt_on_failure: Whether to halt cascade on stage failure
            on_stage_complete: Callback after each stage
            on_cascade_complete: Callback after cascade completes
            silent: Suppress console output
        """
        self.name = name
        self.mode = mode
        self.max_amplification = max_amplification
        self.halt_on_failure = halt_on_failure
        self.on_stage_complete = on_stage_complete
        self.on_cascade_complete = on_cascade_complete
        self.silent = silent

        self._stages: list[CascadeStage] = []
        self._lock = threading.Lock()

        # Statistics
        self._runs_count = 0
        self._successful_runs = 0
        self._failed_runs = 0
        self._total_amplification = 0.0
        self._results_history: list[CascadeResult] = []

    def add_stage(self, stage: CascadeStage) -> 'Cascade':
        """Add a stage to the cascade. Returns self for chaining."""
        self._stages.append(stage)
        if not self.silent:
            print(f"🔗 [Cascade:{self.name}] Added stage: {stage.name}")
        return self

    def insert_stage(self, index: int, stage: CascadeStage) -> 'Cascade':
        """Insert a stage at a specific position."""
        self._stages.insert(index, stage)
        return self

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name."""
        for i, stage in enumerate(self._stages):
            if stage.name == name:
                self._stages.pop(i)
                return True
        return False

    def run(self, input_signal: Any) -> CascadeResult:
        """
        Run the cascade on an input signal.

        Args:
            input_signal: The initial signal to process

        Returns:
            CascadeResult with final output and stage details
        """
        self._runs_count += 1
        start_time = time.time()

        if not self.silent:
            print(f"\n🌊 [Cascade:{self.name}] Starting with {len(self._stages)} stages")

        stage_results: list[StageResult] = []
        current_signal = input_signal
        cumulative_amplification = 1.0
        blocked_at: str | None = None

        for i, stage in enumerate(self._stages):
            stage_start = time.time()

            if not self.silent:
                print(f"  📍 Stage {i+1}/{len(self._stages)}: {stage.name}")

            # Check checkpoint gate
            if stage.checkpoint:
                try:
                    if not stage.checkpoint(current_signal):
                        if not self.silent:
                            print(f"  🚫 Gate blocked at {stage.name}")

                        stage_result = StageResult(
                            stage_name=stage.name,
                            status=StageStatus.BLOCKED,
                            input_signal=current_signal,
                            output_signal=None,
                            processing_time_ms=(time.time() - stage_start) * 1000
                        )
                        stage_results.append(stage_result)
                        blocked_at = stage.name

                        if self.halt_on_failure:
                            break
                        continue
                except Exception as e:
                    if not self.silent:
                        print(f"  ⚠️ Gate error at {stage.name}: {e}")
                    if self.halt_on_failure:
                        stage_result = StageResult(
                            stage_name=stage.name,
                            status=StageStatus.FAILED,
                            input_signal=current_signal,
                            output_signal=None,
                            error=str(e),
                            processing_time_ms=(time.time() - stage_start) * 1000
                        )
                        stage_results.append(stage_result)
                        blocked_at = stage.name
                        break

            # Process stage
            try:
                output_signal = stage.processor(current_signal)

                # Apply amplification
                amplification = stage.amplification
                cumulative_amplification *= amplification

                # Clamp to max
                if cumulative_amplification > self.max_amplification:
                    cumulative_amplification = self.max_amplification
                    if not self.silent:
                        print(f"  ⚡ Amplification clamped to {self.max_amplification}")

                stage_result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.COMPLETED,
                    input_signal=current_signal,
                    output_signal=output_signal,
                    amplification_factor=amplification,
                    processing_time_ms=(time.time() - stage_start) * 1000
                )
                stage_results.append(stage_result)

                if self.on_stage_complete:
                    self.on_stage_complete(stage_result)

                current_signal = output_signal

            except Exception as e:
                if not self.silent:
                    print(f"  💥 Error at {stage.name}: {e}")

                # Try error handler
                if stage.on_error:
                    try:
                        recovery_signal = stage.on_error(e)
                        stage_result = StageResult(
                            stage_name=stage.name,
                            status=StageStatus.COMPLETED,
                            input_signal=current_signal,
                            output_signal=recovery_signal,
                            error=f"Recovered: {e}",
                            processing_time_ms=(time.time() - stage_start) * 1000
                        )
                        stage_results.append(stage_result)
                        current_signal = recovery_signal
                        continue
                    except Exception as recovery_error:
                        if not self.silent:
                            print(f"  💥 Recovery failed: {recovery_error}")

                stage_result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    input_signal=current_signal,
                    output_signal=None,
                    error=str(e),
                    processing_time_ms=(time.time() - stage_start) * 1000
                )
                stage_results.append(stage_result)

                if self.halt_on_failure and stage.required:
                    blocked_at = stage.name
                    break
                elif not stage.required:
                    # Skip non-required stages on failure
                    stage_result.status = StageStatus.SKIPPED

        # Calculate results
        completed_stages = sum(1 for r in stage_results if r.status == StageStatus.COMPLETED)
        success = completed_stages == len(self._stages) and blocked_at is None

        total_time = (time.time() - start_time) * 1000

        result = CascadeResult(
            success=success,
            final_output=current_signal if success else None,
            stages_completed=completed_stages,
            stages_total=len(self._stages),
            total_amplification=cumulative_amplification,
            total_time_ms=total_time,
            stage_results=stage_results,
            blocked_at=blocked_at
        )

        # Update statistics
        if success:
            self._successful_runs += 1
        else:
            self._failed_runs += 1
        self._total_amplification += cumulative_amplification

        # Record history
        self._results_history.append(result)
        if len(self._results_history) > 1000:
            self._results_history = self._results_history[-1000:]

        if self.on_cascade_complete:
            self.on_cascade_complete(result)

        if not self.silent:
            status = "✅" if success else "❌"
            print(f"{status} [Cascade:{self.name}] Completed: {completed_stages}/{len(self._stages)} stages, "
                  f"amplification: {cumulative_amplification:.2f}x")

        return result

    def run_parallel(self, input_signal: Any) -> CascadeResult:
        """
        Run all stages in parallel (fork pattern).

        Each stage receives the same input, results are collected.
        """
        import concurrent.futures

        self._runs_count += 1
        start_time = time.time()

        if not self.silent:
            print(f"\n🌊 [Cascade:{self.name}] Parallel execution with {len(self._stages)} stages")

        stage_results: list[StageResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._stages)) as executor:
            futures = {}
            for stage in self._stages:
                future = executor.submit(self._run_single_stage, stage, input_signal)
                futures[future] = stage.name

            for future in concurrent.futures.as_completed(futures):
                stage_name = futures[future]
                try:
                    result = future.result()
                    stage_results.append(result)
                except Exception as e:
                    stage_results.append(StageResult(
                        stage_name=stage_name,
                        status=StageStatus.FAILED,
                        input_signal=input_signal,
                        output_signal=None,
                        error=str(e)
                    ))

        completed = sum(1 for r in stage_results if r.status == StageStatus.COMPLETED)
        success = completed == len(self._stages)

        # Combine outputs
        outputs = [r.output_signal for r in stage_results if r.status == StageStatus.COMPLETED]
        final_output = outputs if outputs else None

        total_time = (time.time() - start_time) * 1000

        result = CascadeResult(
            success=success,
            final_output=final_output,
            stages_completed=completed,
            stages_total=len(self._stages),
            total_amplification=1.0,
            total_time_ms=total_time,
            stage_results=stage_results
        )

        if success:
            self._successful_runs += 1
        else:
            self._failed_runs += 1

        self._results_history.append(result)

        return result

    def _run_single_stage(self, stage: CascadeStage, input_signal: Any) -> StageResult:
        """Run a single stage."""
        start_time = time.time()

        try:
            output = stage.processor(input_signal)
            return StageResult(
                stage_name=stage.name,
                status=StageStatus.COMPLETED,
                input_signal=input_signal,
                output_signal=output,
                amplification_factor=stage.amplification,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                input_signal=input_signal,
                output_signal=None,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def get_statistics(self) -> dict:
        """Get cascade statistics."""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "stages_count": len(self._stages),
            "stage_names": [s.name for s in self._stages],
            "runs_count": self._runs_count,
            "successful_runs": self._successful_runs,
            "failed_runs": self._failed_runs,
            "success_rate": self._successful_runs / max(1, self._runs_count),
            "average_amplification": self._total_amplification / max(1, self._runs_count),
        }

    def get_history(self, limit: int = 100) -> list[CascadeResult]:
        """Get recent cascade results."""
        return self._results_history[-limit:]


class AgentCascade(Cascade):
    """
    Cascade using BioAgents as processors.

    Each stage is a BioAgent that processes signals.
    """

    def __init__(
        self,
        name: str,
        budget: ATP_Store,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.budget = budget
        self._agents: list[BioAgent] = []

    def add_agent_stage(
        self,
        agent_name: str,
        role: str = "Processor",
        amplification: float = 1.0,
        checkpoint: Callable[[Any], bool] | None = None
    ) -> 'AgentCascade':
        """Add a BioAgent as a cascade stage."""
        agent = BioAgent(agent_name, role, self.budget)
        self._agents.append(agent)

        def agent_processor(signal: Any) -> Any:
            if isinstance(signal, Signal):
                result = agent.express(signal)
            else:
                result = agent.express(Signal(content=str(signal)))
            return result.payload

        stage = CascadeStage(
            name=agent_name,
            processor=agent_processor,
            amplification=amplification,
            checkpoint=checkpoint
        )
        return self.add_stage(stage)


class MAPKCascade(Cascade):
    """
    MAPK-like three-tier signaling cascade.

    Implements the classic MAPKKK → MAPKK → MAPK pattern
    where each tier amplifies the signal.
    """

    def __init__(
        self,
        name: str = "MAPK",
        tier1_amplification: float = 10.0,
        tier2_amplification: float = 10.0,
        tier3_amplification: float = 10.0,
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # Add three tiers
        self.add_stage(CascadeStage(
            name="MAPKKK",
            processor=lambda x: {"signal": x, "tier": 1, "active": True},
            amplification=tier1_amplification
        ))

        self.add_stage(CascadeStage(
            name="MAPKK",
            processor=lambda x: {**x, "tier": 2} if x.get("active") else x,
            amplification=tier2_amplification,
            checkpoint=lambda x: x.get("active", False)
        ))

        self.add_stage(CascadeStage(
            name="MAPK",
            processor=lambda x: {**x, "tier": 3, "response": "ACTIVATED"} if x.get("active") else x,
            amplification=tier3_amplification,
            checkpoint=lambda x: x.get("tier") == 2
        ))

        if not self.silent:
            print(f"🧬 [MAPK Cascade] Initialized with amplification: "
                  f"{tier1_amplification}x → {tier2_amplification}x → {tier3_amplification}x")
