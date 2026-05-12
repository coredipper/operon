import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from operon_ai.state.telomere import Telomere, LifecyclePhase, SenescenceReason, TelomereStatus

def test_telomere_initialization():
    telomere = Telomere(
        max_operations=5000,
        max_lifetime_hours=12,
        idle_timeout_minutes=30,
        error_threshold=50
    )
    assert telomere.max_operations == 5000
    assert telomere.error_threshold == 50
    assert telomere.get_phase() == LifecyclePhase.NASCENT
    assert telomere.is_active() is False
    assert telomere._telomere_length == 5000

def test_telomere_start():
    telomere = Telomere()
    telomere.start()
    assert telomere.get_phase() == LifecyclePhase.ACTIVE
    assert telomere.is_active() is True
    assert telomere._started_at is not None
    assert telomere._last_activity is not None

def test_telomere_tick_depletion():
    telomere = Telomere(max_operations=5, silent=True)
    telomere.start()

    for i in range(4):
        assert telomere.tick() is True
        assert telomere.get_phase() == LifecyclePhase.ACTIVE

    assert telomere.tick() is False
    assert telomere.get_phase() == LifecyclePhase.SENESCENT
    status = telomere.get_status()
    assert status.senescence_reason == SenescenceReason.TELOMERE_DEPLETION

def test_telomere_record_error_accumulation():
    telomere = Telomere(error_threshold=3, silent=True)
    telomere.start()
    # Mocking operations_count to avoid hitting error_rate check early
    telomere._operations_count = 100

    assert telomere.record_error() is True
    assert telomere.record_error() is True
    assert telomere.record_error() is False
    assert telomere.get_phase() == LifecyclePhase.SENESCENT
    status = telomere.get_status()
    assert status.senescence_reason == SenescenceReason.ERROR_ACCUMULATION

def test_telomere_check_timeouts_max_lifetime():
    telomere = Telomere(max_lifetime_hours=1, silent=True)
    telomere.start()

    # Manually backdate started_at
    telomere._started_at = datetime.now() - timedelta(hours=2)
    assert telomere.check_timeouts() is False
    assert telomere.get_phase() == LifecyclePhase.SENESCENT
    status = telomere.get_status()
    assert status.senescence_reason == SenescenceReason.TIMEOUT

def test_telomere_check_timeouts_idle():
    telomere = Telomere(idle_timeout_minutes=10, silent=True)
    telomere.start()

    # Manually backdate last_activity
    telomere._last_activity = datetime.now() - timedelta(minutes=20)
    assert telomere.check_timeouts() is False
    assert telomere.get_phase() == LifecyclePhase.SENESCENT
    status = telomere.get_status()
    assert status.senescence_reason == SenescenceReason.IDLE_TIMEOUT

def test_telomere_heartbeat():
    telomere = Telomere()
    telomere.start()
    old_activity = telomere._last_activity

    # simulate some time passed
    import time
    time.sleep(0.01)

    telomere.heartbeat()
    assert telomere._last_activity > old_activity

def test_telomere_renew():
    telomere = Telomere(max_operations=10, silent=True, allow_renewal=True)
    telomere.start()

    # Deplete
    for _ in range(10):
        telomere.tick()

    assert telomere.get_phase() == LifecyclePhase.SENESCENT

    # Renew
    assert telomere.renew(amount=5) is True
    assert telomere.get_phase() == LifecyclePhase.ACTIVE
    assert telomere._telomere_length == 5

def test_telomere_renew_not_allowed():
    telomere = Telomere(max_operations=10, silent=True, allow_renewal=False)
    telomere.start()

    assert telomere.renew(amount=5) is False

def test_telomere_trigger_apoptosis():
    telomere = Telomere(silent=True)
    telomere.start()
    telomere.trigger_apoptosis("test")
    assert telomere.get_phase() == LifecyclePhase.APOPTOTIC

def test_telomere_terminate():
    telomere = Telomere(silent=True)
    telomere.start()
    telomere.terminate()
    assert telomere.get_phase() == LifecyclePhase.TERMINATED

def test_telomere_get_status():
    telomere = Telomere(max_operations=100)
    telomere.start()
    telomere.tick(cost=10)

    status = telomere.get_status()
    assert isinstance(status, TelomereStatus)
    assert status.phase == LifecyclePhase.ACTIVE
    assert status.telomere_length == 90
    assert status.max_telomere_length == 100
    assert status.health_score > 0

def test_callbacks():
    mock_phase_change = MagicMock()
    mock_senescence = MagicMock()

    telomere = Telomere(
        max_operations=5,
        on_phase_change=mock_phase_change,
        on_senescence=mock_senescence,
        silent=True
    )

    # Test phase change callback on start
    telomere.start()
    mock_phase_change.assert_called_with(LifecyclePhase.NASCENT, LifecyclePhase.ACTIVE)

    # Test senescence callback
    for _ in range(5):
        telomere.tick()

    mock_phase_change.assert_called_with(LifecyclePhase.ACTIVE, LifecyclePhase.SENESCENT)
    mock_senescence.assert_called_with(SenescenceReason.TELOMERE_DEPLETION)
