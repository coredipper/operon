from dataclasses import dataclass, field
from typing import Dict, Optional, Generic, TypeVar, Any

T = TypeVar('T')

@dataclass
class Signal:
    """
    Represents an input stimulus (Transcription Factor).
    """
    content: str
    source: str = "User"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionProtein:
    """
    Represents the expressed output of an agent.
    """
    action_type: str  # e.g., "EXECUTE", "BLOCK", "PERMIT", "FAILURE"
    payload: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FoldedProtein(Generic[T]):
    """
    Represents output that has been validated by a Chaperone.
    """
    valid: bool
    structure: Optional[T] = None
    raw_peptide_chain: str = ""
    error_trace: Optional[str] = None
