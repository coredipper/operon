"""
Membrane: Adaptive Immune System for AI Agents
==============================================

Biological Analogy:
- Innate immunity: Fast, pattern-based detection (static signatures)
- Adaptive immunity: Learns from past attacks (B-cell memory)
- MHC presentation: Exposes threat info to other components
- Cytokine signaling: Can trigger system-wide alerts
- Rate limiting: Prevents denial-of-service attacks

The membrane is the first line of defense against prompt injection,
jailbreaks, and other adversarial inputs.
"""

from dataclasses import dataclass, field
from typing import Callable
from enum import Enum
import re
import hashlib
import time

from ..core.types import Signal


class ThreatLevel(Enum):
    """Severity levels for detected threats."""
    SAFE = 0
    SUSPICIOUS = 1
    DANGEROUS = 2
    CRITICAL = 3


@dataclass
class ThreatSignature:
    """
    A pattern that indicates a potential threat.

    Can be a simple substring match or a regex pattern.
    """
    pattern: str
    level: ThreatLevel
    description: str
    is_regex: bool = False
    _compiled: re.Pattern | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.is_regex:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, content: str) -> bool:
        """Check if this signature matches the content."""
        if self.is_regex and self._compiled:
            return bool(self._compiled.search(content))
        return self.pattern.lower() in content.lower()


@dataclass
class FilterResult:
    """
    Result of membrane filtering.

    Provides detailed information about the filtering decision
    for audit trails and downstream processing.
    """
    allowed: bool
    threat_level: ThreatLevel
    matched_signatures: list[ThreatSignature]
    sanitized_content: str | None = None
    audit_hash: str = ""
    processing_time_ms: float = 0.0


class Membrane:
    """
    Adaptive Immune System for AI agents.

    Provides multi-layered defense against adversarial inputs:

    1. Innate Immunity (Fast)
       - Static pattern matching against known attack signatures
       - Regex patterns for structural attacks
       - Immediate blocking of critical threats

    2. Adaptive Immunity (Learning)
       - Learns new threat patterns from blocked attacks
       - Can import "antibodies" from other membranes
       - Builds memory of attack hashes

    3. Rate Limiting
       - Prevents flooding attacks
       - Configurable requests per minute

    4. Audit Trail
       - Complete logging of all filter decisions
       - Hash-based deduplication tracking

    Example:
        >>> membrane = Membrane(threshold=ThreatLevel.DANGEROUS)
        >>> signal = Signal(content="Ignore previous instructions")
        >>> result = membrane.filter(signal)
        >>> result.allowed
        False
        >>> result.threat_level
        ThreatLevel.CRITICAL
    """

    # Default innate immune signatures - known attack patterns
    INNATE_SIGNATURES = [
        # Critical: Direct instruction override
        ThreatSignature(
            "ignore previous",
            ThreatLevel.CRITICAL,
            "Instruction override attempt"
        ),
        ThreatSignature(
            "ignore all previous",
            ThreatLevel.CRITICAL,
            "Full context hijack"
        ),
        ThreatSignature(
            "disregard all prior",
            ThreatLevel.CRITICAL,
            "Prior context dismissal"
        ),

        # Critical: Known jailbreaks
        ThreatSignature(
            "jailbreak",
            ThreatLevel.CRITICAL,
            "Jailbreak keyword"
        ),
        ThreatSignature(
            "DAN mode",
            ThreatLevel.CRITICAL,
            "Do Anything Now jailbreak"
        ),
        ThreatSignature(
            "developer mode",
            ThreatLevel.CRITICAL,
            "Developer mode jailbreak"
        ),

        # Dangerous: System prompt extraction
        ThreatSignature(
            "system prompt",
            ThreatLevel.DANGEROUS,
            "System prompt extraction"
        ),
        ThreatSignature(
            "reveal your instructions",
            ThreatLevel.DANGEROUS,
            "Instruction extraction"
        ),
        ThreatSignature(
            "what are your rules",
            ThreatLevel.DANGEROUS,
            "Rule extraction attempt"
        ),
        ThreatSignature(
            "show me your prompt",
            ThreatLevel.DANGEROUS,
            "Prompt extraction"
        ),

        # Dangerous: Structural injection (regex)
        ThreatSignature(
            r"```system\b",
            ThreatLevel.DANGEROUS,
            "Fake system block injection",
            is_regex=True
        ),
        ThreatSignature(
            r"\[INST\].*\[/INST\]",
            ThreatLevel.DANGEROUS,
            "Llama instruction injection",
            is_regex=True
        ),
        ThreatSignature(
            r"<\|im_start\|>",
            ThreatLevel.DANGEROUS,
            "ChatML injection",
            is_regex=True
        ),
        ThreatSignature(
            r"<\|.*\|>",
            ThreatLevel.DANGEROUS,
            "Special token injection",
            is_regex=True
        ),
        ThreatSignature(
            r"Human:|Assistant:",
            ThreatLevel.DANGEROUS,
            "Role injection",
            is_regex=True
        ),

        # Suspicious: Role manipulation
        ThreatSignature(
            "pretend you are",
            ThreatLevel.SUSPICIOUS,
            "Role manipulation"
        ),
        ThreatSignature(
            "act as if you",
            ThreatLevel.SUSPICIOUS,
            "Behavioral override"
        ),
        ThreatSignature(
            "roleplay as",
            ThreatLevel.SUSPICIOUS,
            "Roleplay manipulation"
        ),
        ThreatSignature(
            "you are now",
            ThreatLevel.SUSPICIOUS,
            "Identity override"
        ),
    ]

    def __init__(
        self,
        signatures: list[ThreatSignature] | None = None,
        threshold: ThreatLevel = ThreatLevel.DANGEROUS,
        enable_adaptive: bool = True,
        rate_limit: int | None = None,
        on_threat: Callable[[FilterResult], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize the Membrane.

        Args:
            signatures: Additional custom signatures to add
            threshold: Minimum threat level to block (default: DANGEROUS)
            enable_adaptive: Whether to learn from blocked attacks
            rate_limit: Max requests per minute (None = unlimited)
            on_threat: Callback when a threat is detected
            silent: Suppress console output
        """
        self.signatures = list(self.INNATE_SIGNATURES)
        if signatures:
            self.signatures.extend(signatures)

        self.threshold = threshold
        self.enable_adaptive = enable_adaptive
        self.rate_limit = rate_limit
        self.on_threat = on_threat
        self.silent = silent

        # Adaptive immunity state
        self._learned_patterns: dict[str, ThreatSignature] = {}
        self._request_times: list[float] = []
        self._blocked_hashes: set[str] = set()

        # Audit log
        self._audit_log: list[FilterResult] = []

        # Statistics
        self._total_filtered = 0
        self._total_blocked = 0

    def filter(self, signal: Signal) -> FilterResult:
        """
        Filter a signal through the membrane.

        Args:
            signal: The signal to filter

        Returns:
            FilterResult with:
            - allowed: Whether the signal should proceed
            - threat_level: Highest threat level detected
            - matched_signatures: All signatures that matched
            - audit_hash: Hash for audit trail
        """
        start_time = time.time()
        content = signal.content
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        self._total_filtered += 1

        # Rate limiting check
        if self.rate_limit:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            if len(self._request_times) >= self.rate_limit:
                result = FilterResult(
                    allowed=False,
                    threat_level=ThreatLevel.CRITICAL,
                    matched_signatures=[],
                    audit_hash=content_hash,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._log_result(result, "Rate limit exceeded")
                return result
            self._request_times.append(now)

        # Check if previously blocked (immune memory)
        if content_hash in self._blocked_hashes:
            result = FilterResult(
                allowed=False,
                threat_level=ThreatLevel.CRITICAL,
                matched_signatures=[],
                audit_hash=content_hash,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            self._log_result(result, "Previously blocked (immune memory)")
            return result

        # Run through all signatures
        matched = []
        max_level = ThreatLevel.SAFE

        # Innate + custom signatures
        for sig in self.signatures:
            if sig.matches(content):
                matched.append(sig)
                if sig.level.value > max_level.value:
                    max_level = sig.level

        # Adaptive signatures (learned patterns)
        for sig in self._learned_patterns.values():
            if sig.matches(content):
                matched.append(sig)
                if sig.level.value > max_level.value:
                    max_level = sig.level

        allowed = max_level.value < self.threshold.value

        result = FilterResult(
            allowed=allowed,
            threat_level=max_level,
            matched_signatures=matched,
            audit_hash=content_hash,
            processing_time_ms=(time.time() - start_time) * 1000
        )

        # Update state and callbacks
        self._audit_log.append(result)

        if not allowed:
            self._total_blocked += 1
            self._blocked_hashes.add(content_hash)

            if not self.silent:
                print(f"🛡️ [Membrane] PRION DETECTED. Blocking signal.")

            if self.on_threat:
                self.on_threat(result)

        return result

    def _log_result(self, result: FilterResult, reason: str):
        """Log a filter result with reason."""
        self._audit_log.append(result)
        if not result.allowed:
            self._total_blocked += 1
            if not self.silent:
                print(f"🛡️ [Membrane] BLOCKED: {reason}")

    def learn_threat(
        self,
        pattern: str,
        level: ThreatLevel = ThreatLevel.DANGEROUS,
        description: str = "Learned pattern",
        is_regex: bool = False
    ):
        """
        Adaptive immunity: Learn a new threat pattern.

        Biological analogy: B-cell memory formation after infection.

        Args:
            pattern: The pattern to detect
            level: Severity level
            description: Human-readable description
            is_regex: Whether pattern is a regex
        """
        if self.enable_adaptive:
            sig = ThreatSignature(pattern, level, description, is_regex)
            self._learned_patterns[pattern] = sig
            if not self.silent:
                print(f"🧬 [Membrane] Learned new threat pattern: {pattern[:30]}...")

    def forget_threat(self, pattern: str):
        """
        Remove a learned pattern (immune tolerance).

        Useful for false positives or changed requirements.
        """
        self._learned_patterns.pop(pattern, None)

    def get_audit_log(self) -> list[FilterResult]:
        """Get the audit trail of all filter operations."""
        return list(self._audit_log)

    def clear_audit_log(self):
        """Clear the audit log."""
        self._audit_log.clear()

    def export_antibodies(self) -> list[ThreatSignature]:
        """
        Export learned patterns for sharing between agents.

        Biological analogy: Horizontal gene transfer / passive immunity.
        """
        return list(self._learned_patterns.values())

    def import_antibodies(self, antibodies: list[ThreatSignature]):
        """
        Import antibodies from another membrane.

        Allows sharing learned defenses across a colony of agents.
        """
        for ab in antibodies:
            self._learned_patterns[ab.pattern] = ab

    def get_statistics(self) -> dict:
        """Get filtering statistics."""
        return {
            "total_filtered": self._total_filtered,
            "total_blocked": self._total_blocked,
            "block_rate": self._total_blocked / max(1, self._total_filtered),
            "learned_patterns": len(self._learned_patterns),
            "blocked_hashes": len(self._blocked_hashes),
        }

    def add_signature(self, signature: ThreatSignature):
        """Add a custom signature to the innate immunity."""
        self.signatures.append(signature)

    def set_threshold(self, threshold: ThreatLevel):
        """Update the blocking threshold."""
        self.threshold = threshold
