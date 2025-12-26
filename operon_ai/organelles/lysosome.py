"""
Lysosome: Cellular Digestion and Recycling
==========================================

Biological Analogy:
- Autophagy: Breaking down old/damaged cellular components
- Phagocytosis: Digesting external material
- Exocytosis: Expelling waste products
- pH regulation: Maintaining optimal conditions
- Enzyme compartmentalization: Isolated breakdown processes

The Lysosome handles cleanup, recycling, and disposal of:
- Failed operations and their artifacts
- Expired or stale data
- Resources from terminated agents
- Misfolded proteins that couldn't be repaired

This is the garbage collector and janitor of the cellular system.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import weakref
import threading
import time
import logging


_logger = logging.getLogger(__name__)


class WasteType(Enum):
    """Types of cellular waste the lysosome can process."""
    MISFOLDED_PROTEIN = "misfolded"      # Failed parsing/validation
    EXPIRED_CACHE = "expired"             # Stale cached data
    FAILED_OPERATION = "failed_op"        # Operation that errored
    ORPHANED_RESOURCE = "orphaned"        # Resource with no owner
    TOXIC_BYPRODUCT = "toxic"             # Dangerous/sensitive data


@dataclass
class Waste:
    """
    A piece of cellular waste to be processed.

    Contains the waste item plus metadata for proper disposal.
    """
    waste_type: WasteType
    content: Any
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher = more urgent
    metadata: dict = field(default_factory=dict)


@dataclass
class DigestResult:
    """Result of digesting waste."""
    success: bool
    recycled: dict[str, Any] = field(default_factory=dict)
    disposed: int = 0
    errors: list[str] = field(default_factory=list)


class Lysosome:
    """
    Cellular Digestion: Cleans up and recycles waste.

    The Lysosome is responsible for:

    1. Waste Collection
       - Accept waste items from other organelles
       - Categorize by type and priority
       - Queue for processing

    2. Digestion
       - Break down complex waste into recyclable components
       - Extract useful data from failed operations
       - Sanitize sensitive information before disposal

    3. Recycling
       - Return useful components to the cell
       - Update statistics and metrics
       - Feed learnings back to other organelles

    4. Disposal
       - Permanently remove toxic/sensitive data
       - Clear memory and resources
       - Log disposal for audit trails

    5. Autophagy (Self-cleaning)
       - Periodically clean old entries
       - Prevent memory leaks
       - Maintain healthy operation

    Example:
        >>> lysosome = Lysosome()
        >>> waste = Waste(
        ...     waste_type=WasteType.FAILED_OPERATION,
        ...     content={"error": "timeout", "input": "query"},
        ...     source="agent_1"
        ... )
        >>> lysosome.ingest(waste)
        >>> result = lysosome.digest()
        >>> result.success
        True
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        auto_digest_threshold: int = 100,
        retention_hours: float = 24.0,
        digesters: dict[WasteType, Callable[[Waste], dict]] | None = None,
        on_toxic: Callable[[Waste], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Lysosome.

        Args:
            max_queue_size: Maximum waste items to hold
            auto_digest_threshold: Trigger auto-digest when queue reaches this
            retention_hours: How long to keep waste before forced disposal
            digesters: Custom digestion functions per waste type
            on_toxic: Callback for toxic waste (sensitive data)
            silent: Suppress console output
        """
        self.max_queue_size = max_queue_size
        self.auto_digest_threshold = auto_digest_threshold
        self.retention_period = timedelta(hours=retention_hours)
        self.on_toxic = on_toxic
        self.silent = silent

        # Waste queue
        self._queue: list[Waste] = []
        self._lock = threading.Lock()

        # Custom digesters
        self._digesters: dict[WasteType, Callable[[Waste], dict]] = {
            WasteType.MISFOLDED_PROTEIN: self._digest_misfolded,
            WasteType.EXPIRED_CACHE: self._digest_expired,
            WasteType.FAILED_OPERATION: self._digest_failed_op,
            WasteType.ORPHANED_RESOURCE: self._digest_orphaned,
            WasteType.TOXIC_BYPRODUCT: self._digest_toxic,
        }
        if digesters:
            self._digesters.update(digesters)

        # Statistics
        self._total_ingested = 0
        self._total_digested = 0
        self._total_recycled = 0
        self._by_type: dict[WasteType, int] = {t: 0 for t in WasteType}

        # Recycling bin (extracted useful data)
        self._recycling_bin: dict[str, Any] = {}

    def ingest(self, waste: Waste):
        """
        Ingest waste into the lysosome for processing.

        Args:
            waste: The waste item to process
        """
        with self._lock:
            # Check queue capacity
            if len(self._queue) >= self.max_queue_size:
                # Emergency digest
                self._emergency_digest()

            self._queue.append(waste)
            self._total_ingested += 1
            self._by_type[waste.waste_type] += 1

            if not self.silent:
                print(f"🗑️ [Lysosome] Ingested {waste.waste_type.value} from {waste.source}")

            # Auto-digest if threshold reached
            if len(self._queue) >= self.auto_digest_threshold:
                self._auto_digest()

    def ingest_error(
        self,
        error: Exception,
        source: str = "",
        context: dict | None = None
    ):
        """
        Convenience method to ingest an error.

        Args:
            error: The exception that occurred
            source: Where the error came from
            context: Additional context about the error
        """
        waste = Waste(
            waste_type=WasteType.FAILED_OPERATION,
            content={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            },
            source=source,
        )
        self.ingest(waste)

    def ingest_sensitive(self, data: Any, source: str = ""):
        """
        Ingest sensitive data for secure disposal.

        Args:
            data: Sensitive data to dispose of
            source: Where the data came from
        """
        waste = Waste(
            waste_type=WasteType.TOXIC_BYPRODUCT,
            content=data,
            source=source,
            priority=10,  # High priority
        )
        self.ingest(waste)

    def digest(self, max_items: int | None = None) -> DigestResult:
        """
        Process queued waste items.

        Args:
            max_items: Maximum items to process (None = all)

        Returns:
            DigestResult with recycled components and statistics
        """
        with self._lock:
            items_to_process = self._queue[:max_items] if max_items else self._queue[:]
            self._queue = self._queue[len(items_to_process):] if max_items else []

        recycled: dict[str, Any] = {}
        errors: list[str] = []
        disposed = 0

        for waste in items_to_process:
            try:
                digester = self._digesters.get(waste.waste_type, self._digest_default)
                result = digester(waste)

                if result:
                    recycled.update(result)
                    self._total_recycled += 1

                disposed += 1
                self._total_digested += 1

            except Exception as e:
                errors.append(f"Failed to digest {waste.waste_type.value}: {e}")

        # Store recycled materials
        self._recycling_bin.update(recycled)

        if not self.silent and disposed > 0:
            print(f"♻️ [Lysosome] Digested {disposed} items, recycled {len(recycled)} components")

        return DigestResult(
            success=len(errors) == 0,
            recycled=recycled,
            disposed=disposed,
            errors=errors
        )

    def autophagy(self) -> int:
        """
        Self-cleaning: Remove old waste that's past retention.

        Returns:
            Number of items removed
        """
        now = datetime.now()
        removed = 0

        with self._lock:
            original_count = len(self._queue)
            self._queue = [
                w for w in self._queue
                if now - w.created_at < self.retention_period
            ]
            removed = original_count - len(self._queue)

        if removed > 0 and not self.silent:
            print(f"🧹 [Lysosome] Autophagy removed {removed} expired items")

        return removed

    def get_recycled(self, key: str | None = None) -> Any:
        """
        Retrieve recycled components.

        Args:
            key: Specific key to retrieve (None = all)

        Returns:
            Recycled data
        """
        if key:
            return self._recycling_bin.get(key)
        return dict(self._recycling_bin)

    def clear_recycling_bin(self):
        """Clear the recycling bin."""
        self._recycling_bin.clear()

    def _digest_misfolded(self, waste: Waste) -> dict:
        """Process misfolded protein waste."""
        content = waste.content
        recycled = {}

        # Extract useful debugging info
        if isinstance(content, dict):
            if 'raw_input' in content:
                recycled['last_failed_input'] = content['raw_input'][:200]
            if 'error' in content:
                recycled['last_parse_error'] = str(content['error'])[:200]

        return recycled

    def _digest_expired(self, waste: Waste) -> dict:
        """Process expired cache waste."""
        # Just dispose, nothing to recycle
        return {}

    def _digest_failed_op(self, waste: Waste) -> dict:
        """Process failed operation waste."""
        content = waste.content
        recycled = {}

        if isinstance(content, dict):
            # Track error patterns
            error_type = content.get('error_type', 'unknown')
            recycled[f'error_count_{error_type}'] = recycled.get(f'error_count_{error_type}', 0) + 1

            # Extract learnable info
            if 'context' in content:
                recycled['last_failure_context'] = content['context']

        return recycled

    def _digest_orphaned(self, waste: Waste) -> dict:
        """Process orphaned resource waste."""
        # Clean up resources
        content = waste.content

        # If it has a cleanup method, call it
        if hasattr(content, 'cleanup'):
            try:
                content.cleanup()
            except Exception as e:
                _logger.warning(f"Cleanup failed for {waste.waste_type.value}: {e}")

        return {}

    def _digest_toxic(self, waste: Waste) -> dict:
        """Process toxic/sensitive waste."""
        # Notify callback if set
        if self.on_toxic:
            self.on_toxic(waste)

        # Securely dispose - don't recycle anything
        # In a real implementation, you might want to:
        # - Overwrite memory
        # - Log to secure audit trail
        # - Notify security systems

        return {}

    def _digest_default(self, waste: Waste) -> dict:
        """Default digestion for unknown waste types."""
        return {}

    def _emergency_digest(self):
        """Emergency digest when queue is full."""
        # Process oldest 50% of queue
        items_to_process = len(self._queue) // 2
        if items_to_process > 0:
            for waste in self._queue[:items_to_process]:
                try:
                    digester = self._digesters.get(waste.waste_type, self._digest_default)
                    digester(waste)
                    self._total_digested += 1
                except Exception as e:
                    _logger.warning(f"Emergency digest failed for item: {e}")
                    continue  # Continue processing other items
            self._queue = self._queue[items_to_process:]

            if not self.silent:
                print(f"⚠️ [Lysosome] Emergency digest: processed {items_to_process} items")

    def _auto_digest(self):
        """Auto-triggered digest when threshold reached."""
        if not self.silent:
            print(f"[Lysosome] Auto-digesting {len(self._queue)} items")
        # Process half the queue
        self.digest(max_items=len(self._queue) // 2)

    def get_statistics(self) -> dict:
        """Get lysosome statistics."""
        return {
            "queue_size": len(self._queue),
            "total_ingested": self._total_ingested,
            "total_digested": self._total_digested,
            "total_recycled": self._total_recycled,
            "by_type": {t.value: c for t, c in self._by_type.items()},
            "recycling_bin_size": len(self._recycling_bin),
        }

    def get_queue_status(self) -> dict:
        """Get current queue status."""
        with self._lock:
            by_type = {}
            for waste in self._queue:
                t = waste.waste_type.value
                by_type[t] = by_type.get(t, 0) + 1

            return {
                "size": len(self._queue),
                "capacity": self.max_queue_size,
                "utilization": len(self._queue) / self.max_queue_size,
                "by_type": by_type,
            }
