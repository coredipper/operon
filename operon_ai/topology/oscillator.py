"""
Oscillator: Periodic Task and Rhythm Management
================================================

Biological Analogy:
- Circadian rhythm: ~24-hour biological clock
- Cell cycle: Periodic division phases (G1, S, G2, M)
- Heartbeat: Regular cardiac contractions
- Oscillating genes: Genes with periodic expression (like segmentation)
- Ultradian rhythms: Shorter cycles (like hormonal pulses)
- Negative feedback oscillator: Repressilator pattern

The Oscillator topology enables periodic task execution with
configurable rhythms, phase tracking, and synchronization.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import threading
import time
import math


class OscillatorState(Enum):
    """Current state of the oscillator."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    PHASE_TRANSITION = "phase_transition"


class WaveformType(Enum):
    """Type of oscillation waveform."""
    SINE = "sine"           # Smooth sinusoidal
    SQUARE = "square"       # On/off binary
    SAWTOOTH = "sawtooth"   # Linear ramp
    TRIANGLE = "triangle"   # Linear up and down
    PULSE = "pulse"         # Brief spikes


@dataclass
class OscillatorPhase:
    """Definition of an oscillator phase."""
    name: str
    duration_seconds: float
    action: Callable[[], Any] | None = None
    on_enter: Callable[[], None] | None = None
    on_exit: Callable[[], None] | None = None
    skippable: bool = False


@dataclass
class CycleResult:
    """Result from one complete oscillator cycle."""
    cycle_number: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    phases_completed: int
    phase_results: list[dict] = field(default_factory=list)
    interrupted: bool = False


@dataclass
class OscillatorStatus:
    """Current oscillator status."""
    state: OscillatorState
    current_phase: str | None
    phase_progress: float  # 0-1 progress through current phase
    cycle_count: int
    total_runtime_seconds: float
    current_amplitude: float
    frequency_hz: float


class Oscillator:
    """
    Periodic Task Executor with Phase Management.

    Implements biological rhythm patterns for scheduled,
    repeating tasks with configurable phases and waveforms.

    Features:

    1. Multi-Phase Cycles
       - Define phases within each cycle
       - Per-phase actions and callbacks
       - Automatic phase transitions

    2. Waveform Patterns
       - SINE: Smooth oscillation
       - SQUARE: Binary on/off
       - SAWTOOTH: Ramp patterns
       - TRIANGLE: Linear oscillation
       - PULSE: Brief activity spikes

    3. Amplitude Control
       - Dynamic amplitude adjustment
       - Dampening over time
       - External amplitude modulation

    4. Frequency Management
       - Configurable frequency
       - Frequency modulation
       - Period tracking

    5. Synchronization
       - Sync multiple oscillators
       - Phase locking
       - Beat frequency detection

    Example:
        >>> osc = Oscillator(frequency_hz=0.1)  # 10 second period
        >>> osc.add_phase(OscillatorPhase(
        ...     name="active",
        ...     duration_seconds=5,
        ...     action=lambda: print("Active!")
        ... ))
        >>> osc.add_phase(OscillatorPhase(
        ...     name="rest",
        ...     duration_seconds=5,
        ...     action=lambda: print("Resting...")
        ... ))
        >>> osc.start()
    """

    def __init__(
        self,
        frequency_hz: float = 1.0,
        amplitude: float = 1.0,
        waveform: WaveformType = WaveformType.SINE,
        max_cycles: int | None = None,
        damping_factor: float = 0.0,
        on_cycle_complete: Callable[[CycleResult], None] | None = None,
        on_tick: Callable[[float], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Oscillator.

        Args:
            frequency_hz: Oscillation frequency in Hz
            amplitude: Initial amplitude (0-1)
            waveform: Type of waveform pattern
            max_cycles: Maximum cycles before stopping (None = infinite)
            damping_factor: Amplitude decay per cycle (0 = no decay)
            on_cycle_complete: Callback after each cycle
            on_tick: Callback on each tick with current value
            silent: Suppress console output
        """
        self.frequency_hz = frequency_hz
        self.amplitude = amplitude
        self.initial_amplitude = amplitude
        self.waveform = waveform
        self.max_cycles = max_cycles
        self.damping_factor = damping_factor
        self.on_cycle_complete = on_cycle_complete
        self.on_tick = on_tick
        self.silent = silent

        # Phases
        self._phases: list[OscillatorPhase] = []
        self._current_phase_index = 0

        # State
        self._state = OscillatorState.STOPPED
        self._cycle_count = 0
        self._start_time: datetime | None = None
        self._phase_start_time: datetime | None = None

        # Threading
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # Statistics
        self._total_runtime = 0.0
        self._cycle_history: list[CycleResult] = []

    @property
    def period_seconds(self) -> float:
        """Get the oscillation period in seconds."""
        return 1.0 / self.frequency_hz if self.frequency_hz > 0 else float('inf')

    def add_phase(self, phase: OscillatorPhase) -> 'Oscillator':
        """Add a phase to the oscillator cycle. Returns self for chaining."""
        self._phases.append(phase)
        if not self.silent:
            print(f"🔄 [Oscillator] Added phase: {phase.name} ({phase.duration_seconds}s)")
        return self

    def set_frequency(self, frequency_hz: float):
        """Set the oscillation frequency."""
        with self._lock:
            self.frequency_hz = frequency_hz
            if not self.silent:
                print(f"🔄 [Oscillator] Frequency: {frequency_hz} Hz (period: {self.period_seconds:.2f}s)")

    def set_amplitude(self, amplitude: float):
        """Set the oscillation amplitude."""
        with self._lock:
            self.amplitude = max(0, min(1, amplitude))

    def start(self):
        """Start the oscillator."""
        if self._state == OscillatorState.RUNNING:
            return

        self._state = OscillatorState.RUNNING
        self._start_time = datetime.now()
        self._stop_event.clear()
        self._pause_event.set()  # Not paused

        if not self.silent:
            print(f"▶️ [Oscillator] Started at {self.frequency_hz} Hz")

        # Start background thread
        self._thread = threading.Thread(target=self._oscillate, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the oscillator."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused

        if self._thread:
            self._thread.join(timeout=2.0)

        self._state = OscillatorState.STOPPED

        if self._start_time:
            self._total_runtime += (datetime.now() - self._start_time).total_seconds()

        if not self.silent:
            print(f"⏹️ [Oscillator] Stopped after {self._cycle_count} cycles")

    def pause(self):
        """Pause the oscillator."""
        self._pause_event.clear()
        self._state = OscillatorState.PAUSED
        if not self.silent:
            print("⏸️ [Oscillator] Paused")

    def resume(self):
        """Resume a paused oscillator."""
        if self._state == OscillatorState.PAUSED:
            self._pause_event.set()
            self._state = OscillatorState.RUNNING
            if not self.silent:
                print("▶️ [Oscillator] Resumed")

    def reset(self):
        """Reset the oscillator to initial state."""
        was_running = self._state == OscillatorState.RUNNING
        if was_running:
            self.stop()

        with self._lock:
            self._cycle_count = 0
            self._current_phase_index = 0
            self.amplitude = self.initial_amplitude
            self._total_runtime = 0.0
            self._cycle_history.clear()

        if was_running:
            self.start()

    def _oscillate(self):
        """Main oscillation loop."""
        cycle_start = datetime.now()
        phase_results: list[dict] = []

        while not self._stop_event.is_set():
            # Check pause
            self._pause_event.wait()

            if self._stop_event.is_set():
                break

            # Check max cycles
            if self.max_cycles and self._cycle_count >= self.max_cycles:
                if not self.silent:
                    print(f"🏁 [Oscillator] Max cycles ({self.max_cycles}) reached")
                break

            # Run phases if defined
            if self._phases:
                self._run_phases(phase_results)
            else:
                # Simple tick-based oscillation
                self._simple_oscillate()

            # Complete cycle
            cycle_end = datetime.now()
            cycle_duration = (cycle_end - cycle_start).total_seconds()

            result = CycleResult(
                cycle_number=self._cycle_count,
                start_time=cycle_start,
                end_time=cycle_end,
                duration_seconds=cycle_duration,
                phases_completed=len(phase_results),
                phase_results=phase_results.copy()
            )

            self._cycle_history.append(result)
            if len(self._cycle_history) > 1000:
                self._cycle_history = self._cycle_history[-1000:]

            if self.on_cycle_complete:
                self.on_cycle_complete(result)

            # Apply damping
            if self.damping_factor > 0:
                self.amplitude *= (1 - self.damping_factor)
                if self.amplitude < 0.01:
                    if not self.silent:
                        print("🔇 [Oscillator] Amplitude damped to zero")
                    break

            self._cycle_count += 1
            phase_results.clear()
            cycle_start = datetime.now()

        self._state = OscillatorState.STOPPED

    def _run_phases(self, phase_results: list[dict]):
        """Run through all phases."""
        for i, phase in enumerate(self._phases):
            if self._stop_event.is_set():
                return

            self._current_phase_index = i
            self._phase_start_time = datetime.now()
            self._state = OscillatorState.PHASE_TRANSITION

            if not self.silent:
                print(f"  🔄 Phase: {phase.name}")

            # On enter callback
            if phase.on_enter:
                try:
                    phase.on_enter()
                except Exception as e:
                    if not self.silent:
                        print(f"  ⚠️ Phase enter error: {e}")

            self._state = OscillatorState.RUNNING

            # Run phase action periodically during phase duration
            phase_start = time.time()
            while time.time() - phase_start < phase.duration_seconds:
                if self._stop_event.is_set():
                    return

                # Check pause
                self._pause_event.wait()

                if phase.action:
                    try:
                        result = phase.action()
                        phase_results.append({
                            "phase": phase.name,
                            "result": result,
                            "timestamp": datetime.now()
                        })
                    except Exception as e:
                        if not self.silent:
                            print(f"  ⚠️ Phase action error: {e}")

                # Calculate current waveform value and notify
                elapsed = time.time() - phase_start
                progress = elapsed / phase.duration_seconds
                current_value = self._calculate_waveform(progress) * self.amplitude

                if self.on_tick:
                    self.on_tick(current_value)

                # Small sleep to prevent busy waiting
                time.sleep(0.1)

            # On exit callback
            if phase.on_exit:
                try:
                    phase.on_exit()
                except Exception as e:
                    if not self.silent:
                        print(f"  ⚠️ Phase exit error: {e}")

    def _simple_oscillate(self):
        """Simple oscillation without phases."""
        period = self.period_seconds
        start = time.time()

        while time.time() - start < period:
            if self._stop_event.is_set():
                return

            self._pause_event.wait()

            elapsed = time.time() - start
            progress = elapsed / period
            current_value = self._calculate_waveform(progress) * self.amplitude

            if self.on_tick:
                self.on_tick(current_value)

            time.sleep(0.05)

    def _calculate_waveform(self, phase: float) -> float:
        """Calculate waveform value at given phase (0-1)."""
        if self.waveform == WaveformType.SINE:
            return math.sin(2 * math.pi * phase)

        elif self.waveform == WaveformType.SQUARE:
            return 1.0 if phase < 0.5 else -1.0

        elif self.waveform == WaveformType.SAWTOOTH:
            return 2.0 * phase - 1.0

        elif self.waveform == WaveformType.TRIANGLE:
            if phase < 0.5:
                return 4.0 * phase - 1.0
            else:
                return 3.0 - 4.0 * phase

        elif self.waveform == WaveformType.PULSE:
            return 1.0 if phase < 0.1 else 0.0

        return 0.0

    def get_current_value(self) -> float:
        """Get current oscillation value."""
        if self._state != OscillatorState.RUNNING or not self._start_time:
            return 0.0

        elapsed = (datetime.now() - self._start_time).total_seconds()
        phase = (elapsed * self.frequency_hz) % 1.0
        return self._calculate_waveform(phase) * self.amplitude

    def get_status(self) -> OscillatorStatus:
        """Get current oscillator status."""
        current_phase = None
        phase_progress = 0.0

        if self._phases and 0 <= self._current_phase_index < len(self._phases):
            current_phase = self._phases[self._current_phase_index].name
            if self._phase_start_time:
                elapsed = (datetime.now() - self._phase_start_time).total_seconds()
                phase_duration = self._phases[self._current_phase_index].duration_seconds
                phase_progress = min(1.0, elapsed / phase_duration)

        total_runtime = self._total_runtime
        if self._start_time and self._state == OscillatorState.RUNNING:
            total_runtime += (datetime.now() - self._start_time).total_seconds()

        return OscillatorStatus(
            state=self._state,
            current_phase=current_phase,
            phase_progress=phase_progress,
            cycle_count=self._cycle_count,
            total_runtime_seconds=total_runtime,
            current_amplitude=self.amplitude,
            frequency_hz=self.frequency_hz
        )

    def get_statistics(self) -> dict:
        """Get oscillator statistics."""
        avg_cycle_duration = 0.0
        if self._cycle_history:
            avg_cycle_duration = sum(c.duration_seconds for c in self._cycle_history) / len(self._cycle_history)

        return {
            "state": self._state.value,
            "frequency_hz": self.frequency_hz,
            "period_seconds": self.period_seconds,
            "amplitude": self.amplitude,
            "waveform": self.waveform.value,
            "cycle_count": self._cycle_count,
            "phases_count": len(self._phases),
            "total_runtime_seconds": self._total_runtime,
            "average_cycle_duration": avg_cycle_duration,
        }

    def get_history(self, limit: int = 100) -> list[CycleResult]:
        """Get recent cycle history."""
        return self._cycle_history[-limit:]


class CircadianOscillator(Oscillator):
    """
    24-hour circadian rhythm oscillator.

    Pre-configured with day/night phases for biological timing.
    """

    def __init__(
        self,
        day_action: Callable[[], Any] | None = None,
        night_action: Callable[[], Any] | None = None,
        day_hours: float = 16.0,
        night_hours: float = 8.0,
        **kwargs
    ):
        # 24 hour period = 1/86400 Hz (one cycle per day)
        super().__init__(
            frequency_hz=1.0 / (day_hours + night_hours) / 3600,
            waveform=WaveformType.SINE,
            **kwargs
        )

        self.add_phase(OscillatorPhase(
            name="day",
            duration_seconds=day_hours * 3600,
            action=day_action,
            on_enter=lambda: print("☀️ Day phase") if not self.silent else None,
            on_exit=lambda: print("🌅 Dusk") if not self.silent else None
        ))

        self.add_phase(OscillatorPhase(
            name="night",
            duration_seconds=night_hours * 3600,
            action=night_action,
            on_enter=lambda: print("🌙 Night phase") if not self.silent else None,
            on_exit=lambda: print("🌄 Dawn") if not self.silent else None
        ))


class HeartbeatOscillator(Oscillator):
    """
    Heartbeat-style oscillator for health checks and keepalives.

    Fast, regular pulses for system monitoring.
    """

    def __init__(
        self,
        beats_per_minute: float = 60.0,
        on_beat: Callable[[], Any] | None = None,
        **kwargs
    ):
        super().__init__(
            frequency_hz=beats_per_minute / 60.0,
            waveform=WaveformType.PULSE,
            **kwargs
        )

        self._on_beat = on_beat

        # Single beat phase
        period = 60.0 / beats_per_minute
        self.add_phase(OscillatorPhase(
            name="beat",
            duration_seconds=period * 0.1,  # Brief contraction
            action=on_beat
        ))
        self.add_phase(OscillatorPhase(
            name="rest",
            duration_seconds=period * 0.9  # Rest between beats
        ))


class CellCycleOscillator(Oscillator):
    """
    Cell cycle oscillator with G1, S, G2, M phases.

    Simulates the biological cell division cycle.
    """

    def __init__(
        self,
        cycle_duration_hours: float = 24.0,
        on_g1: Callable[[], Any] | None = None,
        on_s: Callable[[], Any] | None = None,
        on_g2: Callable[[], Any] | None = None,
        on_m: Callable[[], Any] | None = None,
        **kwargs
    ):
        super().__init__(
            frequency_hz=1.0 / (cycle_duration_hours * 3600),
            waveform=WaveformType.SAWTOOTH,
            **kwargs
        )

        cycle_seconds = cycle_duration_hours * 3600

        # Typical cell cycle phases
        self.add_phase(OscillatorPhase(
            name="G1",
            duration_seconds=cycle_seconds * 0.4,  # 40% - Growth
            action=on_g1,
            on_enter=lambda: print("🌱 G1: Growth phase") if not self.silent else None
        ))

        self.add_phase(OscillatorPhase(
            name="S",
            duration_seconds=cycle_seconds * 0.3,  # 30% - DNA Synthesis
            action=on_s,
            on_enter=lambda: print("🧬 S: DNA Synthesis") if not self.silent else None
        ))

        self.add_phase(OscillatorPhase(
            name="G2",
            duration_seconds=cycle_seconds * 0.2,  # 20% - Preparation
            action=on_g2,
            on_enter=lambda: print("⚡ G2: Preparation") if not self.silent else None
        ))

        self.add_phase(OscillatorPhase(
            name="M",
            duration_seconds=cycle_seconds * 0.1,  # 10% - Mitosis
            action=on_m,
            on_enter=lambda: print("💫 M: Mitosis") if not self.silent else None
        ))
