"""
Autophagy Daemon: Cognitive Self-Healing through Context Pruning
================================================================

Biological Analogy:
- Autophagy: Cells "eating themselves" - digesting accumulated waste (damaged
  organelles, protein aggregates) to recycle components and prevent toxic buildup.
- Sleep/Wake Cycles: During sleep, the brain consolidates important memories
  and clears metabolic waste (glymphatic system).
- Synaptic Pruning: Unused neural connections are eliminated to maintain
  efficient information processing.

The Autophagy Daemon extends basic memory management into proactive health
maintenance:
1. Monitor context window utilization (like cellular "toxicity sensors")
2. Detect when context is becoming "polluted" (too much noise, failed attempts)
3. Trigger a "sleep cycle" - pause execution to consolidate
4. Summarize useful state into long-term memory (HistoneStore)
5. Flush raw context via Lysosome
6. Agent wakes with clean window plus summary

The key insight: Context pollution degrades performance. Proactive cleanup
prevents gradual degradation before it becomes catastrophic.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from enum import Enum

from ..state.histone import HistoneStore, MarkerType, MarkerStrength
from ..organelles.lysosome import Lysosome, Waste, WasteType


class ContextHealthStatus(Enum):
    """Health status of the context window."""

    HEALTHY = "healthy"  # Plenty of room
    ACCUMULATING = "accumulating"  # Building up, watch it
    CRITICAL = "critical"  # Needs pruning soon
    PRUNING = "pruning"  # Currently in sleep/prune cycle


@dataclass
class ContextMetrics:
    """Metrics about current context state."""

    estimated_tokens: int
    max_tokens: int
    fill_percentage: float
    status: ContextHealthStatus
    useful_content_ratio: float  # Estimated ratio of useful vs noise
    last_pruned: datetime | None = None


@dataclass
class PruneResult:
    """Result of a context pruning operation."""

    pruned: bool
    tokens_before: int
    tokens_after: int
    tokens_freed: int
    summary_stored: str
    waste_items_flushed: int
    duration_ms: float


@dataclass
class AutophagyDaemon:
    """
    Background monitor that triggers context consolidation.

    Biological parallel: Autophagy - cells digesting their own accumulated
    waste to recycle resources and stay healthy. Like the brain's glymphatic
    system that clears waste during sleep.

    The daemon monitors context window size and triggers "sleep cycles" when
    pollution exceeds threshold:
    1. Pause execution (sleep phase)
    2. Summarize useful state
    3. Store summary in long-term memory
    4. Flush raw context
    5. Resume with clean window + summary

    Example:
        >>> daemon = AutophagyDaemon(
        ...     histone_store=HistoneStore(),
        ...     lysosome=Lysosome(),
        ...     summarizer=my_summarizer,
        ...     toxicity_threshold=0.8,
        ... )
        >>> new_context, pruned = daemon.check_and_prune(
        ...     context="... very long context ...",
        ...     max_tokens=8000,
        ... )
        >>> if pruned:
        ...     print("Context was pruned and consolidated")
    """

    histone_store: HistoneStore
    lysosome: Lysosome
    summarizer: Callable[[str], str]
    toxicity_threshold: float = 0.8  # Prune at 80% fill
    warning_threshold: float = 0.6  # Warn at 60% fill
    min_tokens_for_pruning: int = 1000  # Don't prune tiny contexts
    tokens_per_char: float = 0.25  # Rough estimate (4 chars per token)
    silent: bool = False

    # Tracking
    _prune_count: int = field(default=0, init=False)
    _total_tokens_freed: int = field(default=0, init=False)
    _last_check: datetime | None = field(default=None, init=False)

    def __post_init__(self):
        self._prune_count = 0
        self._total_tokens_freed = 0
        self._last_check = None

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple character-based heuristic. Real implementation
        should use the actual tokenizer for the model being used.
        """
        return int(len(text) * self.tokens_per_char)

    def assess_health(self, context: str, max_tokens: int) -> ContextMetrics:
        """
        Assess health of current context.

        Args:
            context: The current context window contents
            max_tokens: Maximum tokens the context can hold

        Returns:
            ContextMetrics with health assessment
        """
        estimated = self.estimate_tokens(context)
        fill = estimated / max_tokens if max_tokens > 0 else 1.0

        # Determine status
        if fill >= self.toxicity_threshold:
            status = ContextHealthStatus.CRITICAL
        elif fill >= self.warning_threshold:
            status = ContextHealthStatus.ACCUMULATING
        else:
            status = ContextHealthStatus.HEALTHY

        # Estimate useful content ratio (heuristic based on patterns)
        useful_ratio = self._estimate_useful_ratio(context)

        return ContextMetrics(
            estimated_tokens=estimated,
            max_tokens=max_tokens,
            fill_percentage=fill,
            status=status,
            useful_content_ratio=useful_ratio,
        )

    def check_and_prune(
        self,
        context: str,
        max_tokens: int,
        force: bool = False,
    ) -> tuple[str, PruneResult | None]:
        """
        Check context health and prune if needed.

        This is the main entry point for the autophagy cycle.

        Args:
            context: Current context window contents
            max_tokens: Maximum context window size
            force: Force pruning regardless of threshold

        Returns:
            Tuple of (new_context, PruneResult if pruned else None)
        """
        import time

        self._last_check = datetime.now(timezone.utc)
        metrics = self.assess_health(context, max_tokens)

        # Decide if pruning is needed
        should_prune = (
            force
            or metrics.status == ContextHealthStatus.CRITICAL
            or (
                metrics.status == ContextHealthStatus.ACCUMULATING
                and metrics.useful_content_ratio < 0.5
            )
        )

        # Don't prune tiny contexts
        if metrics.estimated_tokens < self.min_tokens_for_pruning:
            should_prune = False

        if not should_prune:
            if not self.silent and metrics.status == ContextHealthStatus.ACCUMULATING:
                print(
                    f"🧹 [Autophagy] Context at {metrics.fill_percentage:.0%} "
                    f"(threshold: {self.toxicity_threshold:.0%}). Monitoring..."
                )
            return context, None

        # Trigger pruning
        start = time.time()

        if not self.silent:
            print(
                f"🧹 [Autophagy] SLEEP CYCLE - Context at {metrics.fill_percentage:.0%}. "
                f"Initiating consolidation..."
            )

        # Step 1: Summarize useful content
        summary = self.summarizer(context)

        if not self.silent:
            print(f"🧹 [Autophagy] Summarized {metrics.estimated_tokens} tokens → {self.estimate_tokens(summary)} tokens")

        # Step 2: Store summary in long-term memory
        marker_hash = self.histone_store.add_marker(
            content=summary,
            marker_type=MarkerType.ACETYLATION,  # Semi-permanent
            strength=MarkerStrength.MODERATE,
            tags=["autophagy", "context_summary"],
            context="Consolidated from context pruning",
        )

        # Step 3: Flush raw context via Lysosome
        waste = Waste(
            waste_type=WasteType.EXPIRED_CACHE,
            content={"context": context, "summary_hash": marker_hash},
            source="autophagy_daemon",
            metadata={"tokens_freed": metrics.estimated_tokens},
        )
        self.lysosome.ingest(waste)

        # Step 4: Construct new clean context
        new_context = self._construct_clean_context(summary)

        duration = (time.time() - start) * 1000
        tokens_after = self.estimate_tokens(new_context)
        tokens_freed = metrics.estimated_tokens - tokens_after

        self._prune_count += 1
        self._total_tokens_freed += tokens_freed

        if not self.silent:
            print(
                f"🧹 [Autophagy] WAKE - Freed {tokens_freed} tokens. "
                f"Context now at {tokens_after / max_tokens:.0%}"
            )

        return new_context, PruneResult(
            pruned=True,
            tokens_before=metrics.estimated_tokens,
            tokens_after=tokens_after,
            tokens_freed=tokens_freed,
            summary_stored=summary,
            waste_items_flushed=1,
            duration_ms=duration,
        )

    def _estimate_useful_ratio(self, context: str) -> float:
        """
        Estimate ratio of useful content vs noise.

        Uses simple heuristics - real implementation could use
        semantic analysis.
        """
        # Markers of noise/waste in context
        noise_markers = [
            "Error:",
            "Failed:",
            "Traceback",
            "Exception",
            "FAILED",
            "retry",
            "timeout",
            "Invalid",
            "Unable to",
            "I apologize",
            "I cannot",
        ]

        lines = context.split("\n")
        if not lines:
            return 1.0

        noise_lines = sum(
            1 for line in lines if any(marker in line for marker in noise_markers)
        )

        return 1.0 - (noise_lines / len(lines))

    def _construct_clean_context(self, summary: str) -> str:
        """
        Construct the clean context to return after pruning.

        Combines the summary with a marker indicating consolidation occurred.
        """
        return (
            f"[Context consolidated via autophagy]\n"
            f"Summary of previous context:\n{summary}\n"
            f"[End of consolidated context]\n"
        )

    def stats(self) -> dict:
        """Return daemon statistics."""
        return {
            "prune_count": self._prune_count,
            "total_tokens_freed": self._total_tokens_freed,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "average_tokens_per_prune": (
                self._total_tokens_freed / self._prune_count
                if self._prune_count > 0
                else 0
            ),
        }


def create_simple_summarizer(max_summary_lines: int = 10) -> Callable[[str], str]:
    """
    Create a simple summarizer that extracts key lines.

    In a real implementation, this would use an LLM to intelligently
    summarize the context.
    """

    def summarizer(context: str) -> str:
        lines = context.split("\n")

        # Filter out noise
        noise_markers = [
            "Error:",
            "Failed:",
            "Traceback",
            "Exception",
            "retry",
            "timeout",
        ]

        useful_lines = [
            line
            for line in lines
            if line.strip()
            and not any(marker in line for marker in noise_markers)
        ]

        # Take first and last useful lines (most likely to be important)
        if len(useful_lines) <= max_summary_lines:
            selected = useful_lines
        else:
            half = max_summary_lines // 2
            selected = useful_lines[:half] + useful_lines[-half:]

        return "\n".join(selected)

    return summarizer
