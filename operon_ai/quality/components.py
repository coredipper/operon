# operon_ai/quality/components.py
"""Components for the proteasome system."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from operon_ai.core.types import DataType
from .types import UbiquitinTag


@dataclass
class ProvenanceContext:
    """Runtime context available during tag processing."""

    tag: UbiquitinTag
    source_module: str
    target_module: str
    source_reliability: float = 1.0
    system_load: float = 0.0
    operation_criticality: str = "normal"
    data_type: Optional[DataType] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class E3Ligase:
    """
    Context-sensitive tagger — reduces confidence.

    Biological parallel: E3 ubiquitin ligases that attach
    ubiquitin chains to proteins based on quality signals.
    """

    name: str
    active: Callable[[ProvenanceContext], bool]
    substrate_match: Callable[[Any], bool]
    tag_strength: Callable[[ProvenanceContext], float]  # Multiplier (0.0-1.0)


@dataclass
class Deubiquitinase:
    """
    Context-sensitive eraser — restores confidence.

    Biological parallel: DUBs that remove ubiquitin chains,
    rescuing proteins from degradation.
    """

    name: str
    active: Callable[[ProvenanceContext], bool]
    rescue_condition: Callable[[UbiquitinTag, ProvenanceContext], bool]
    rescue_amount: float  # Added to confidence


@dataclass
class ChaperoneRepair:
    """
    Attempts data repair before degradation.

    Biological parallel: Chaperones that refold misfolded
    proteins before they are sent to proteasome.
    """

    name: str
    can_repair: Callable[[Any, UbiquitinTag], bool]
    repair: Callable[[Any, UbiquitinTag], tuple[Any, bool]]  # (repaired_data, success)
    confidence_boost: float = 0.3
