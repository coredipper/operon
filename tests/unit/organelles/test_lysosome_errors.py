"""Error handling tests for Lysosome."""
import pytest
from operon_ai import Lysosome, Waste, WasteType


class FailingCleanup:
    """Object that raises on cleanup."""
    def cleanup(self):
        raise RuntimeError("Cleanup failed!")


def test_cleanup_failure_is_logged(caplog):
    """Failed cleanup should be logged, not silently swallowed."""
    import logging
    caplog.set_level(logging.WARNING)

    lysosome = Lysosome(silent=True)  # Silence console output
    waste = Waste(
        waste_type=WasteType.ORPHANED_RESOURCE,  # This type calls cleanup()
        content=FailingCleanup(),
        source="test",
    )
    lysosome.ingest(waste)

    # Digest should not raise, but should log
    result = lysosome.digest()

    # Check that warning was logged
    assert any("cleanup" in record.message.lower() or "failed" in record.message.lower()
               for record in caplog.records), "Cleanup failure should be logged"


def test_emergency_digest_logs_errors(caplog):
    """Emergency digest should log errors, not silently ignore."""
    import logging
    caplog.set_level(logging.WARNING)

    lysosome = Lysosome(max_queue_size=10, silent=True)  # Small queue to trigger emergency

    # Fill with problematic waste to trigger emergency digest
    for i in range(15):
        lysosome.ingest(Waste(
            waste_type=WasteType.ORPHANED_RESOURCE,
            content=FailingCleanup() if i % 3 == 0 else f"data_{i}",
            source="test",
        ))

    # Emergency digest should have been triggered, should not crash
    # Check that errors during emergency digest were logged
    assert any("emergency" in record.message.lower() or "failed" in record.message.lower()
               for record in caplog.records), "Emergency digest failures should be logged"
