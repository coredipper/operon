"""
Healing: Self-Repair Mechanisms for Agentic Systems
===================================================

Biological systems don't just crash when components fail - they repair,
recycle, and regenerate. This module provides three primary healing patterns
that enable agentic systems to maintain homeostasis.

Healing Mechanisms:
    - ChaperoneLoop: Structural healing through error feedback
    - RegenerativeSwarm: Metabolic healing through apoptosis and regeneration
    - AutophagyDaemon: Cognitive healing through context pruning

The key insight: Most agent frameworks focus on Action (doing things).
Operon focuses on Homeostasis (staying alive). By implementing Chaperone
Loops for structural repair and Autophagy for context hygiene, we build
systems that degrade gracefully rather than crashing catastrophically.

Example:
    >>> from operon_ai.healing import ChaperoneLoop, RegenerativeSwarm, AutophagyDaemon
    >>> from operon_ai import Chaperone, HistoneStore, Lysosome
    >>>
    >>> # Structural healing
    >>> loop = ChaperoneLoop(generator=my_llm, chaperone=Chaperone(), schema=MySchema)
    >>> result = loop.heal("Generate output")
    >>>
    >>> # Cognitive healing
    >>> daemon = AutophagyDaemon(
    ...     histone_store=HistoneStore(),
    ...     lysosome=Lysosome(),
    ...     summarizer=my_summarizer,
    ... )
    >>> new_context, pruned = daemon.check_and_prune(context, max_tokens=8000)
"""

from .chaperone_loop import (
    ChaperoneLoop,
    HealingResult,
    HealingOutcome,
    RefoldingAttempt,
    create_mock_healing_generator,
)

from .regenerative_swarm import (
    RegenerativeSwarm,
    SwarmResult,
    SimpleWorker,
    WorkerMemory,
    WorkerStatus,
    ApoptosisEvent,
    ApoptosisReason,
    RegenerationEvent,
    create_default_summarizer,
)

from .autophagy_daemon import (
    AutophagyDaemon,
    PruneResult,
    ContextMetrics,
    ContextHealthStatus,
    create_simple_summarizer,
)

__all__ = [
    # Chaperone Loop (Structural Healing)
    "ChaperoneLoop",
    "HealingResult",
    "HealingOutcome",
    "RefoldingAttempt",
    "create_mock_healing_generator",
    # Regenerative Swarm (Metabolic Healing)
    "RegenerativeSwarm",
    "SwarmResult",
    "SimpleWorker",
    "WorkerMemory",
    "WorkerStatus",
    "ApoptosisEvent",
    "ApoptosisReason",
    "RegenerationEvent",
    "create_default_summarizer",
    # Autophagy Daemon (Cognitive Healing)
    "AutophagyDaemon",
    "PruneResult",
    "ContextMetrics",
    "ContextHealthStatus",
    "create_simple_summarizer",
]
