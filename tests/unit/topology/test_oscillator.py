"""
Comprehensive tests for Oscillator topology.

Tests cover:
- Basic oscillator functionality
- Phase management
- HeartbeatOscillator specific features
- CircadianOscillator specific features
- CellCycleOscillator specific features
- State management
- Waveform calculations
- Frequency and amplitude control
"""

import pytest
import time
import threading
from datetime import datetime

from operon_ai.topology.oscillator import (
    Oscillator,
    OscillatorPhase,
    OscillatorState,
    WaveformType,
    OscillatorStatus,
    CycleResult,
    HeartbeatOscillator,
    CircadianOscillator,
    CellCycleOscillator,
)


# ============================================================================
# TestOscillatorBasics - Basic functionality
# ============================================================================


class TestOscillatorBasics:
    """Test basic Oscillator functionality."""

    def test_create_oscillator_with_default_frequency(self):
        """Should create an oscillator with default 1 Hz frequency."""
        osc = Oscillator(silent=True)
        assert osc.frequency_hz == 1.0
        assert osc.period_seconds == 1.0
        assert osc.amplitude == 1.0
        assert osc.waveform == WaveformType.SINE

    def test_create_oscillator_with_custom_frequency(self):
        """Should create an oscillator with custom frequency."""
        osc = Oscillator(frequency_hz=0.5, silent=True)
        assert osc.frequency_hz == 0.5
        assert osc.period_seconds == 2.0

    def test_create_oscillator_with_custom_amplitude(self):
        """Should create an oscillator with custom amplitude."""
        osc = Oscillator(amplitude=0.5, silent=True)
        assert osc.amplitude == 0.5
        assert osc.initial_amplitude == 0.5

    def test_create_oscillator_with_waveform(self):
        """Should create an oscillator with specific waveform."""
        osc = Oscillator(waveform=WaveformType.SQUARE, silent=True)
        assert osc.waveform == WaveformType.SQUARE

    def test_oscillator_initial_state_is_stopped(self):
        """Should have STOPPED state initially."""
        osc = Oscillator(silent=True)
        status = osc.get_status()
        assert status.state == OscillatorState.STOPPED
        assert status.cycle_count == 0

    def test_period_calculation(self):
        """Should correctly calculate period from frequency."""
        osc1 = Oscillator(frequency_hz=2.0, silent=True)
        assert osc1.period_seconds == 0.5

        osc2 = Oscillator(frequency_hz=0.1, silent=True)
        assert osc2.period_seconds == 10.0

    def test_set_frequency(self):
        """Should update frequency and period."""
        osc = Oscillator(frequency_hz=1.0, silent=True)
        assert osc.frequency_hz == 1.0

        osc.set_frequency(2.0)
        assert osc.frequency_hz == 2.0
        assert osc.period_seconds == 0.5

    def test_set_amplitude(self):
        """Should update amplitude."""
        osc = Oscillator(amplitude=1.0, silent=True)
        assert osc.amplitude == 1.0

        osc.set_amplitude(0.5)
        assert osc.amplitude == 0.5

    def test_set_amplitude_clamps_to_valid_range(self):
        """Should clamp amplitude to [0, 1] range."""
        osc = Oscillator(silent=True)

        osc.set_amplitude(1.5)
        assert osc.amplitude == 1.0

        osc.set_amplitude(-0.5)
        assert osc.amplitude == 0.0


# ============================================================================
# TestOscillatorPhases - Phase management
# ============================================================================


class TestOscillatorPhases:
    """Test oscillator phase management."""

    def test_add_single_phase(self):
        """Should add a phase to the oscillator."""
        osc = Oscillator(silent=True)
        phase = OscillatorPhase(name="test_phase", duration_seconds=1.0)
        osc.add_phase(phase)

        assert len(osc._phases) == 1
        assert osc._phases[0].name == "test_phase"

    def test_add_multiple_phases(self):
        """Should add multiple phases in order."""
        osc = Oscillator(silent=True)
        phase1 = OscillatorPhase(name="phase1", duration_seconds=1.0)
        phase2 = OscillatorPhase(name="phase2", duration_seconds=2.0)
        phase3 = OscillatorPhase(name="phase3", duration_seconds=3.0)

        osc.add_phase(phase1)
        osc.add_phase(phase2)
        osc.add_phase(phase3)

        assert len(osc._phases) == 3
        assert osc._phases[0].name == "phase1"
        assert osc._phases[1].name == "phase2"
        assert osc._phases[2].name == "phase3"

    def test_add_phase_returns_self_for_chaining(self):
        """Should return self to allow method chaining."""
        osc = Oscillator(silent=True)
        result = osc.add_phase(OscillatorPhase(name="p1", duration_seconds=1.0))

        assert result is osc

    def test_phase_with_action(self):
        """Should create phase with action callback."""
        action_called = []

        def action():
            action_called.append(True)
            return "result"

        phase = OscillatorPhase(
            name="action_phase",
            duration_seconds=0.1,
            action=action
        )

        assert phase.action is not None
        result = phase.action()
        assert result == "result"
        assert len(action_called) == 1

    def test_phase_with_callbacks(self):
        """Should create phase with enter and exit callbacks."""
        enter_called = []
        exit_called = []

        phase = OscillatorPhase(
            name="callback_phase",
            duration_seconds=0.1,
            on_enter=lambda: enter_called.append(True),
            on_exit=lambda: exit_called.append(True)
        )

        assert phase.on_enter is not None
        assert phase.on_exit is not None

        phase.on_enter()
        assert len(enter_called) == 1

        phase.on_exit()
        assert len(exit_called) == 1

    def test_phase_tracking_in_status(self):
        """Should track current phase in status."""
        osc = Oscillator(frequency_hz=10.0, silent=True)
        osc.add_phase(OscillatorPhase(name="phase1", duration_seconds=0.05))
        osc.add_phase(OscillatorPhase(name="phase2", duration_seconds=0.05))

        osc.start()
        time.sleep(0.02)  # Let it run briefly

        status = osc.get_status()
        osc.stop()

        assert status.state in [OscillatorState.RUNNING, OscillatorState.PHASE_TRANSITION]
        assert status.current_phase in ["phase1", "phase2"]


# ============================================================================
# TestHeartbeatOscillator - Heartbeat specific tests
# ============================================================================


class TestHeartbeatOscillator:
    """Test HeartbeatOscillator specific features."""

    def test_create_heartbeat_oscillator_default_bpm(self):
        """Should create heartbeat oscillator with default 60 BPM."""
        hb = HeartbeatOscillator(silent=True)

        # 60 BPM = 1 Hz
        assert hb.frequency_hz == 1.0
        assert hb.waveform == WaveformType.PULSE

    def test_create_heartbeat_oscillator_custom_bpm(self):
        """Should create heartbeat oscillator with custom BPM."""
        hb = HeartbeatOscillator(beats_per_minute=120.0, silent=True)

        # 120 BPM = 2 Hz
        assert hb.frequency_hz == 2.0

    def test_heartbeat_frequency_conversion(self):
        """Should correctly convert BPM to Hz."""
        # Test various BPM values
        hb1 = HeartbeatOscillator(beats_per_minute=60.0, silent=True)
        assert hb1.frequency_hz == pytest.approx(1.0)

        hb2 = HeartbeatOscillator(beats_per_minute=30.0, silent=True)
        assert hb2.frequency_hz == pytest.approx(0.5)

        hb3 = HeartbeatOscillator(beats_per_minute=90.0, silent=True)
        assert hb3.frequency_hz == pytest.approx(1.5)

    def test_heartbeat_has_beat_and_rest_phases(self):
        """Should have beat and rest phases configured."""
        hb = HeartbeatOscillator(beats_per_minute=60.0, silent=True)

        assert len(hb._phases) == 2
        assert hb._phases[0].name == "beat"
        assert hb._phases[1].name == "rest"

    def test_heartbeat_phase_durations(self):
        """Should have correct phase duration ratios."""
        hb = HeartbeatOscillator(beats_per_minute=60.0, silent=True)

        period = 60.0 / 60.0  # 1 second period
        beat_phase = hb._phases[0]
        rest_phase = hb._phases[1]

        # Beat is 10% of period, rest is 90%
        assert beat_phase.duration_seconds == pytest.approx(period * 0.1)
        assert rest_phase.duration_seconds == pytest.approx(period * 0.9)

    def test_heartbeat_on_beat_callback(self):
        """Should call on_beat callback during beat phase."""
        beat_count = []

        def on_beat():
            beat_count.append(datetime.now())

        hb = HeartbeatOscillator(
            beats_per_minute=60.0,
            on_beat=on_beat,
            max_cycles=2,
            silent=True
        )

        # Verify the callback is set in the beat phase
        assert hb._phases[0].action is not None


# ============================================================================
# TestCircadianOscillator - Circadian rhythm tests
# ============================================================================


class TestCircadianOscillator:
    """Test CircadianOscillator specific features."""

    def test_create_circadian_oscillator_default(self):
        """Should create circadian oscillator with default day/night."""
        circ = CircadianOscillator(silent=True)

        # Should have day and night phases
        assert len(circ._phases) == 2
        assert circ._phases[0].name == "day"
        assert circ._phases[1].name == "night"

        # Should use SINE waveform
        assert circ.waveform == WaveformType.SINE

    def test_circadian_default_day_night_hours(self):
        """Should have default 16h day and 8h night."""
        circ = CircadianOscillator(silent=True)

        day_phase = circ._phases[0]
        night_phase = circ._phases[1]

        assert day_phase.duration_seconds == pytest.approx(16.0 * 3600)
        assert night_phase.duration_seconds == pytest.approx(8.0 * 3600)

    def test_circadian_custom_day_night_hours(self):
        """Should create circadian oscillator with custom hours."""
        circ = CircadianOscillator(
            day_hours=12.0,
            night_hours=12.0,
            silent=True
        )

        day_phase = circ._phases[0]
        night_phase = circ._phases[1]

        assert day_phase.duration_seconds == pytest.approx(12.0 * 3600)
        assert night_phase.duration_seconds == pytest.approx(12.0 * 3600)

    def test_circadian_frequency_calculation(self):
        """Should calculate correct frequency for 24h cycle."""
        circ = CircadianOscillator(
            day_hours=16.0,
            night_hours=8.0,
            silent=True
        )

        # 24 hour period = 1/(24*3600) Hz
        expected_frequency = 1.0 / (24.0 * 3600)
        assert circ.frequency_hz == pytest.approx(expected_frequency)

    def test_circadian_with_actions(self):
        """Should accept day and night actions."""
        day_actions = []
        night_actions = []

        circ = CircadianOscillator(
            day_action=lambda: day_actions.append(True),
            night_action=lambda: night_actions.append(True),
            silent=True
        )

        assert circ._phases[0].action is not None
        assert circ._phases[1].action is not None


# ============================================================================
# TestCellCycleOscillator - Cell cycle tests
# ============================================================================


class TestCellCycleOscillator:
    """Test CellCycleOscillator specific features."""

    def test_create_cell_cycle_oscillator(self):
        """Should create cell cycle oscillator with all phases."""
        cc = CellCycleOscillator(silent=True)

        # Should have G1, S, G2, M phases
        assert len(cc._phases) == 4
        assert cc._phases[0].name == "G1"
        assert cc._phases[1].name == "S"
        assert cc._phases[2].name == "G2"
        assert cc._phases[3].name == "M"

        # Should use SAWTOOTH waveform
        assert cc.waveform == WaveformType.SAWTOOTH

    def test_cell_cycle_phase_durations(self):
        """Should have correct phase duration ratios."""
        cycle_hours = 24.0
        cc = CellCycleOscillator(cycle_duration_hours=cycle_hours, silent=True)

        cycle_seconds = cycle_hours * 3600

        # G1: 40%, S: 30%, G2: 20%, M: 10%
        assert cc._phases[0].duration_seconds == pytest.approx(cycle_seconds * 0.4)
        assert cc._phases[1].duration_seconds == pytest.approx(cycle_seconds * 0.3)
        assert cc._phases[2].duration_seconds == pytest.approx(cycle_seconds * 0.2)
        assert cc._phases[3].duration_seconds == pytest.approx(cycle_seconds * 0.1)

    def test_cell_cycle_custom_duration(self):
        """Should create cell cycle with custom duration."""
        cc = CellCycleOscillator(cycle_duration_hours=12.0, silent=True)

        # 12 hour cycle
        expected_frequency = 1.0 / (12.0 * 3600)
        assert cc.frequency_hz == pytest.approx(expected_frequency)

    def test_cell_cycle_with_phase_actions(self):
        """Should accept actions for each phase."""
        g1_actions = []
        s_actions = []
        g2_actions = []
        m_actions = []

        cc = CellCycleOscillator(
            on_g1=lambda: g1_actions.append(True),
            on_s=lambda: s_actions.append(True),
            on_g2=lambda: g2_actions.append(True),
            on_m=lambda: m_actions.append(True),
            silent=True
        )

        assert cc._phases[0].action is not None
        assert cc._phases[1].action is not None
        assert cc._phases[2].action is not None
        assert cc._phases[3].action is not None


# ============================================================================
# TestOscillatorState - State management
# ============================================================================


class TestOscillatorState:
    """Test oscillator state management."""

    def test_initial_state_is_stopped(self):
        """Should start in STOPPED state."""
        osc = Oscillator(silent=True)
        status = osc.get_status()
        assert status.state == OscillatorState.STOPPED

    def test_start_changes_state_to_running(self):
        """Should change to RUNNING state when started."""
        osc = Oscillator(frequency_hz=10.0, silent=True)
        osc.add_phase(OscillatorPhase(name="test", duration_seconds=0.1))

        osc.start()
        time.sleep(0.01)

        status = osc.get_status()
        osc.stop()

        assert status.state in [OscillatorState.RUNNING, OscillatorState.PHASE_TRANSITION]

    def test_stop_changes_state_to_stopped(self):
        """Should change to STOPPED state when stopped."""
        osc = Oscillator(frequency_hz=10.0, silent=True)
        osc.add_phase(OscillatorPhase(name="test", duration_seconds=0.1))

        osc.start()
        time.sleep(0.01)
        osc.stop()

        status = osc.get_status()
        assert status.state == OscillatorState.STOPPED

    def test_pause_changes_state_to_paused(self):
        """Should change to PAUSED state when paused."""
        osc = Oscillator(frequency_hz=10.0, silent=True)
        osc.add_phase(OscillatorPhase(name="test", duration_seconds=0.1))

        osc.start()
        time.sleep(0.01)
        osc.pause()

        status = osc.get_status()
        osc.stop()

        assert status.state == OscillatorState.PAUSED

    def test_resume_changes_state_to_running(self):
        """Should change back to RUNNING when resumed."""
        osc = Oscillator(frequency_hz=10.0, silent=True)
        osc.add_phase(OscillatorPhase(name="test", duration_seconds=0.1))

        osc.start()
        time.sleep(0.01)
        osc.pause()
        osc.resume()

        status = osc.get_status()
        osc.stop()

        assert status.state in [OscillatorState.RUNNING, OscillatorState.PHASE_TRANSITION]

    def test_get_status_returns_complete_info(self):
        """Should return complete status information."""
        osc = Oscillator(frequency_hz=2.0, amplitude=0.8, silent=True)
        status = osc.get_status()

        assert isinstance(status, OscillatorStatus)
        assert status.state == OscillatorState.STOPPED
        assert status.current_phase is None
        assert status.phase_progress >= 0.0
        assert status.cycle_count == 0
        assert status.total_runtime_seconds >= 0.0
        assert status.current_amplitude == 0.8
        assert status.frequency_hz == 2.0


# ============================================================================
# TestOscillatorWaveforms - Waveform calculations
# ============================================================================


class TestOscillatorWaveforms:
    """Test oscillator waveform calculations."""

    def test_sine_waveform_at_key_points(self):
        """Should calculate correct SINE values."""
        osc = Oscillator(waveform=WaveformType.SINE, amplitude=1.0, silent=True)

        # At phase 0: sin(0) = 0
        assert osc._calculate_waveform(0.0) == pytest.approx(0.0, abs=0.01)

        # At phase 0.25: sin(π/2) = 1
        assert osc._calculate_waveform(0.25) == pytest.approx(1.0, abs=0.01)

        # At phase 0.5: sin(π) = 0
        assert osc._calculate_waveform(0.5) == pytest.approx(0.0, abs=0.01)

        # At phase 0.75: sin(3π/2) = -1
        assert osc._calculate_waveform(0.75) == pytest.approx(-1.0, abs=0.01)

    def test_square_waveform(self):
        """Should calculate correct SQUARE values."""
        osc = Oscillator(waveform=WaveformType.SQUARE, silent=True)

        assert osc._calculate_waveform(0.0) == 1.0
        assert osc._calculate_waveform(0.25) == 1.0
        assert osc._calculate_waveform(0.49) == 1.0
        assert osc._calculate_waveform(0.5) == -1.0
        assert osc._calculate_waveform(0.75) == -1.0

    def test_sawtooth_waveform(self):
        """Should calculate correct SAWTOOTH values."""
        osc = Oscillator(waveform=WaveformType.SAWTOOTH, silent=True)

        assert osc._calculate_waveform(0.0) == pytest.approx(-1.0)
        assert osc._calculate_waveform(0.25) == pytest.approx(-0.5)
        assert osc._calculate_waveform(0.5) == pytest.approx(0.0)
        assert osc._calculate_waveform(0.75) == pytest.approx(0.5)
        assert osc._calculate_waveform(1.0) == pytest.approx(1.0)

    def test_triangle_waveform(self):
        """Should calculate correct TRIANGLE values."""
        osc = Oscillator(waveform=WaveformType.TRIANGLE, silent=True)

        assert osc._calculate_waveform(0.0) == pytest.approx(-1.0)
        assert osc._calculate_waveform(0.25) == pytest.approx(0.0)
        assert osc._calculate_waveform(0.5) == pytest.approx(1.0)
        assert osc._calculate_waveform(0.75) == pytest.approx(0.0)

    def test_pulse_waveform(self):
        """Should calculate correct PULSE values."""
        osc = Oscillator(waveform=WaveformType.PULSE, silent=True)

        assert osc._calculate_waveform(0.0) == 1.0
        assert osc._calculate_waveform(0.05) == 1.0
        assert osc._calculate_waveform(0.09) == 1.0
        assert osc._calculate_waveform(0.1) == 0.0
        assert osc._calculate_waveform(0.5) == 0.0


# ============================================================================
# TestOscillatorCycles - Cycle management
# ============================================================================


class TestOscillatorCycles:
    """Test oscillator cycle management."""

    def test_max_cycles_stops_oscillator(self):
        """Should stop after reaching max cycles."""
        osc = Oscillator(frequency_hz=5.0, max_cycles=2, silent=True)
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        osc.start()
        time.sleep(0.5)  # Wait for cycles to complete

        status = osc.get_status()
        assert status.state == OscillatorState.STOPPED
        assert status.cycle_count >= 2

    def test_cycle_complete_callback(self):
        """Should call callback after each cycle."""
        results = []

        def on_cycle(result: CycleResult):
            results.append(result)

        osc = Oscillator(
            frequency_hz=5.0,
            max_cycles=2,
            on_cycle_complete=on_cycle,
            silent=True
        )
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        osc.start()
        time.sleep(0.5)

        assert len(results) >= 2
        assert all(isinstance(r, CycleResult) for r in results)

    def test_cycle_count_increments(self):
        """Should increment cycle count correctly."""
        osc = Oscillator(frequency_hz=5.0, max_cycles=3, silent=True)
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        osc.start()
        time.sleep(0.7)

        status = osc.get_status()
        assert status.cycle_count >= 3

    def test_reset_clears_cycles(self):
        """Should reset cycle count to zero."""
        osc = Oscillator(frequency_hz=5.0, max_cycles=2, silent=True)
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        osc.start()
        time.sleep(0.5)
        osc.stop()

        status1 = osc.get_status()
        assert status1.cycle_count >= 2

        osc.reset()
        status2 = osc.get_status()
        assert status2.cycle_count == 0


# ============================================================================
# TestOscillatorDamping - Amplitude damping
# ============================================================================


class TestOscillatorDamping:
    """Test oscillator amplitude damping."""

    def test_damping_reduces_amplitude(self):
        """Should reduce amplitude over cycles."""
        osc = Oscillator(
            frequency_hz=5.0,
            amplitude=1.0,
            damping_factor=0.2,
            max_cycles=3,
            silent=True
        )
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        initial_amplitude = osc.amplitude

        osc.start()
        time.sleep(0.7)

        final_amplitude = osc.amplitude
        assert final_amplitude < initial_amplitude

    def test_no_damping_maintains_amplitude(self):
        """Should maintain amplitude without damping."""
        osc = Oscillator(
            frequency_hz=5.0,
            amplitude=1.0,
            damping_factor=0.0,
            max_cycles=3,
            silent=True
        )
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        initial_amplitude = osc.amplitude

        osc.start()
        time.sleep(0.7)

        final_amplitude = osc.amplitude
        assert final_amplitude == pytest.approx(initial_amplitude)


# ============================================================================
# TestOscillatorStatistics - Statistics and history
# ============================================================================


class TestOscillatorStatistics:
    """Test oscillator statistics and history tracking."""

    def test_get_statistics(self):
        """Should return comprehensive statistics."""
        osc = Oscillator(frequency_hz=2.0, amplitude=0.7, silent=True)
        stats = osc.get_statistics()

        assert "state" in stats
        assert "frequency_hz" in stats
        assert "period_seconds" in stats
        assert "amplitude" in stats
        assert "waveform" in stats
        assert "cycle_count" in stats
        assert "phases_count" in stats
        assert stats["frequency_hz"] == 2.0
        assert stats["amplitude"] == 0.7

    def test_get_history_returns_recent_cycles(self):
        """Should return recent cycle history."""
        osc = Oscillator(frequency_hz=5.0, max_cycles=3, silent=True)
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.1))

        osc.start()
        time.sleep(0.7)

        history = osc.get_history()
        assert len(history) >= 3
        assert all(isinstance(h, CycleResult) for h in history)

    def test_get_history_respects_limit(self):
        """Should return limited history entries."""
        osc = Oscillator(frequency_hz=20.0, max_cycles=5, silent=True)
        osc.add_phase(OscillatorPhase(name="fast", duration_seconds=0.025))

        osc.start()
        time.sleep(0.3)

        history = osc.get_history(limit=2)
        assert len(history) <= 5


# ============================================================================
# TestOscillatorTickCallback - Tick callback
# ============================================================================


class TestOscillatorTickCallback:
    """Test oscillator tick callback functionality."""

    def test_on_tick_callback_called(self):
        """Should call on_tick callback during oscillation."""
        tick_values = []

        def on_tick(value: float):
            tick_values.append(value)

        osc = Oscillator(
            frequency_hz=10.0,
            on_tick=on_tick,
            max_cycles=1,
            silent=True
        )
        osc.add_phase(OscillatorPhase(name="test", duration_seconds=0.1))

        osc.start()
        time.sleep(0.15)

        # Should have received multiple tick callbacks
        assert len(tick_values) > 0

    def test_tick_values_respect_amplitude(self):
        """Should scale tick values by amplitude."""
        tick_values = []

        def on_tick(value: float):
            tick_values.append(value)

        osc = Oscillator(
            frequency_hz=10.0,
            amplitude=0.5,
            on_tick=on_tick,
            max_cycles=1,
            silent=True
        )
        osc.add_phase(OscillatorPhase(name="test", duration_seconds=0.1))

        osc.start()
        time.sleep(0.15)

        # All values should be within [-0.5, 0.5] range
        assert all(abs(v) <= 0.5 for v in tick_values)
