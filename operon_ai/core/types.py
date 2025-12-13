"""
Core Types: Biological Data Structures
======================================

Biological Analogy:
- Signal: Transcription factor / ligand binding to receptor
- ActionProtein: Expressed protein that performs cellular function
- FoldedProtein: Properly folded (validated) protein structure
- CellState: Current state of the cellular machinery

These types form the categorical foundation - objects in our
biological category with morphisms defined by agent processing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Generic, TypeVar, Any, List
from enum import Enum, IntEnum
from datetime import datetime

T = TypeVar('T')


class SignalType(Enum):
    """Types of signals in biological systems."""
    EXTERNAL = "external"       # From outside the cell (user input)
    INTERNAL = "internal"       # From within the cell (inter-agent)
    PARACRINE = "paracrine"     # From nearby cells (local agents)
    ENDOCRINE = "endocrine"     # From distant cells (remote agents)
    AUTOCRINE = "autocrine"     # Self-signaling (feedback)


class SignalStrength(Enum):
    """Strength of the signal."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    SATURATING = 4


@dataclass
class Signal:
    """
    Represents an input stimulus (Transcription Factor).

    Biological Analogy:
    A signal is like a ligand binding to a receptor, triggering
    a cellular response. The content is the message, metadata
    carries additional context about the signaling pathway.

    Attributes:
        content: The message content
        source: Origin of the signal
        signal_type: Type of signaling (external, internal, etc.)
        strength: Signal strength/intensity
        timestamp: When the signal was created
        metadata: Additional context
        trace_id: For distributed tracing
    """
    content: str
    source: str = "User"
    signal_type: SignalType = SignalType.EXTERNAL
    strength: SignalStrength = SignalStrength.MODERATE
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

    def with_metadata(self, **kwargs) -> 'Signal':
        """Create a new signal with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return Signal(
            content=self.content,
            source=self.source,
            signal_type=self.signal_type,
            strength=self.strength,
            timestamp=self.timestamp,
            metadata=new_metadata,
            trace_id=self.trace_id,
        )

    def amplify(self) -> 'Signal':
        """Increase signal strength."""
        strengths = list(SignalStrength)
        idx = min(len(strengths) - 1, strengths.index(self.strength) + 1)
        return Signal(
            content=self.content,
            source=self.source,
            signal_type=self.signal_type,
            strength=strengths[idx],
            timestamp=self.timestamp,
            metadata=self.metadata,
            trace_id=self.trace_id,
        )

    def attenuate(self) -> 'Signal':
        """Decrease signal strength."""
        strengths = list(SignalStrength)
        idx = max(0, strengths.index(self.strength) - 1)
        return Signal(
            content=self.content,
            source=self.source,
            signal_type=self.signal_type,
            strength=strengths[idx],
            timestamp=self.timestamp,
            metadata=self.metadata,
            trace_id=self.trace_id,
        )


class ActionType(Enum):
    """Types of actions an agent can take."""
    EXECUTE = "execute"     # Perform the requested action
    PERMIT = "permit"       # Allow another agent to proceed
    BLOCK = "block"         # Block the action
    DEFER = "defer"         # Defer to another agent
    FAILURE = "failure"     # Action failed
    UNKNOWN = "unknown"     # Unknown/unhandled


class IntegrityLabel(IntEnum):
    """
    Integrity label for information-flow control.

    Mirrors the paper's ordering U ≤ V ≤ T:
    - UNTRUSTED: User input / raw model text
    - VALIDATED: Schema-checked / partially validated
    - TRUSTED: Tool-grounded / trusted boundary
    """

    UNTRUSTED = 0
    VALIDATED = 1
    TRUSTED = 2


class DataType(Enum):
    """
    Abstract data types for wiring interfaces.

    These are intentionally coarse-grained: they describe what kind of
    value flows, not its full Python type.
    """

    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    STOP = "stop"
    APPROVAL = "approval"


class Capability(Enum):
    """
    Capability/effect tags for least-privilege reasoning.

    These model what a module/tool can do (e.g., network, filesystem).
    """

    READ_FS = "read_fs"
    WRITE_FS = "write_fs"
    NET = "net"
    EXEC_CODE = "exec_code"
    MONEY = "money"
    EMAIL_SEND = "email_send"


@dataclass(frozen=True)
class ApprovalToken:
    """
    A proof-carrying approval token used for two-key execution gates.

    In the paper, privileged sinks require an (Approval, TRUSTED) input.
    This token binds an approval decision to a specific request.
    """

    request_hash: str
    issuer: str
    reason: str = ""
    confidence: float = 1.0
    integrity: IntegrityLabel = IntegrityLabel.TRUSTED
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionProtein:
    """
    Represents the expressed output of an agent.

    Biological Analogy:
    The protein is the functional output of gene expression.
    It carries the action to be taken and any associated data.

    Attributes:
        action_type: Type of action (EXECUTE, BLOCK, PERMIT, etc.)
        payload: The action data or message
        confidence: Confidence level (0.0 - 1.0)
        source_agent: Which agent produced this
        timestamp: When the action was generated
        metadata: Additional context
    """
    action_type: str  # Keep as str for backward compatibility
    payload: Any
    confidence: float
    source_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """Check if this represents a successful action."""
        return self.action_type in ("EXECUTE", "PERMIT", "SUCCESS")

    def is_blocking(self) -> bool:
        """Check if this blocks further processing."""
        return self.action_type in ("BLOCK", "FAILURE")

    def with_confidence(self, new_confidence: float) -> 'ActionProtein':
        """Create new protein with adjusted confidence."""
        return ActionProtein(
            action_type=self.action_type,
            payload=self.payload,
            confidence=new_confidence,
            source_agent=self.source_agent,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )


@dataclass
class FoldedProtein(Generic[T]):
    """
    Represents output that has been validated by a Chaperone.

    Biological Analogy:
    Proteins must fold correctly to function. A chaperone assists
    in proper folding and detects misfolded proteins.

    Attributes:
        valid: Whether the protein folded correctly
        structure: The typed structure if valid
        raw_peptide_chain: The raw string before folding
        error_trace: Error message if folding failed
        folding_attempts: Number of attempts needed
    """
    valid: bool
    structure: Optional[T] = None
    raw_peptide_chain: str = ""
    error_trace: Optional[str] = None
    folding_attempts: int = 1

    def map(self, func) -> 'FoldedProtein':
        """Apply a function to the structure if valid."""
        if self.valid and self.structure is not None:
            try:
                new_structure = func(self.structure)
                return FoldedProtein(
                    valid=True,
                    structure=new_structure,
                    raw_peptide_chain=self.raw_peptide_chain,
                    folding_attempts=self.folding_attempts,
                )
            except Exception as e:
                return FoldedProtein(
                    valid=False,
                    structure=None,
                    raw_peptide_chain=self.raw_peptide_chain,
                    error_trace=str(e),
                    folding_attempts=self.folding_attempts,
                )
        return self


@dataclass
class CellState:
    """
    Aggregate state of a cellular agent.

    Tracks the overall health and status of an agent,
    combining metabolic, epigenetic, and lifecycle state.

    Attributes:
        agent_name: Name of the agent
        phase: Current lifecycle phase
        health_score: Overall health (0.0 - 1.0)
        energy_level: Current energy ratio
        memory_count: Number of stored memories
        operations_count: Total operations performed
        errors_count: Total errors encountered
        timestamp: When state was captured
    """
    agent_name: str
    phase: str = "active"
    health_score: float = 1.0
    energy_level: float = 1.0
    memory_count: int = 0
    operations_count: int = 0
    errors_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def is_healthy(self) -> bool:
        """Check if cell is in healthy state."""
        return self.health_score > 0.5 and self.energy_level > 0.1

    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.operations_count == 0:
            return 0.0
        return self.errors_count / self.operations_count


@dataclass
class Pathway:
    """
    Represents a signaling pathway.

    A pathway is a series of transformations that a signal
    undergoes as it moves through the cellular machinery.

    Attributes:
        name: Pathway identifier
        stages: Ordered list of stage names
        current_stage: Current position in pathway
        signals: Signals that have traveled this path
    """
    name: str
    stages: List[str] = field(default_factory=list)
    current_stage: int = 0
    signals: List[Signal] = field(default_factory=list)

    def advance(self) -> bool:
        """Move to next stage. Returns False if at end."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

    def current_stage_name(self) -> Optional[str]:
        """Get name of current stage."""
        if 0 <= self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None

    def is_complete(self) -> bool:
        """Check if pathway is complete."""
        return self.current_stage >= len(self.stages) - 1
