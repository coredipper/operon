import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength, EpigeneticMarker

def test_clear_type():
    histone = HistoneStore(silent=True)

    # Add markers of different types
    id1 = histone.add_marker("test 1", marker_type=MarkerType.METHYLATION)
    id2 = histone.add_marker("test 2", marker_type=MarkerType.ACETYLATION)
    id3 = histone.add_marker("test 3", marker_type=MarkerType.METHYLATION)
    id4 = histone.add_marker("test 4", marker_type=MarkerType.PHOSPHORYLATION)

    assert len(histone._markers) == 4

    # Clear ACETYLATION markers
    histone.clear_type(MarkerType.ACETYLATION)

    assert len(histone._markers) == 3
    assert id2 not in histone._markers
    assert id1 in histone._markers
    assert id3 in histone._markers
    assert id4 in histone._markers

    # Clear METHYLATION markers
    histone.clear_type(MarkerType.METHYLATION)

    assert len(histone._markers) == 1
    assert id1 not in histone._markers
    assert id3 not in histone._markers
    assert id4 in histone._markers


def test_purge_expired():
    histone = HistoneStore(silent=True)

    id1 = histone.add_marker("expired", decay_hours=1.0)
    id2 = histone.add_marker("valid", decay_hours=24.0)

    # Modify created_at to simulate time passing
    # id1 is older than its decay_hours (1.0), so it should expire
    histone._markers[id1].created_at = datetime.now() - timedelta(hours=2)
    # id2 is newer than its decay_hours (24.0), so it should NOT expire
    histone._markers[id2].created_at = datetime.now() - timedelta(hours=2)

    assert len(histone._markers) == 2

    histone._purge_expired()

    assert len(histone._markers) == 1
    assert id1 not in histone._markers
    assert id2 in histone._markers


def test_maybe_check_decay():
    histone = HistoneStore(silent=True)
    histone.decay_check_interval = 5

    id1 = histone.add_marker("test 1", decay_hours=1.0)
    # Make it expired
    histone._markers[id1].created_at = datetime.now() - timedelta(hours=2)

    assert len(histone._markers) == 1

    # _operation_count is typically incremented somewhere.
    histone._operation_count = 4
    histone._maybe_check_decay()
    assert len(histone._markers) == 1 # Not modulo 5

    histone._operation_count = 5
    histone._maybe_check_decay()
    assert len(histone._markers) == 0 # Purged
