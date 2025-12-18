"""MHC Display - behavioral fingerprint collector."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib
import json
import re
import statistics

from .types import MHCPeptide


@dataclass
class Observation:
    """Single observation of agent output."""
    output: Optional[str]
    response_time: float
    confidence: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MHCDisplay:
    """
    Collects behavioral observations and generates fingerprints.

    Like MHC molecules displaying protein fragments, this displays
    statistical signatures of agent behavior for inspection.
    """

    agent_id: str
    window_size: int = 100
    min_observations: int = 10

    observations: list[Observation] = field(default_factory=list)
    canary_results: list[bool] = field(default_factory=list)

    def record(
        self,
        output: Optional[str],
        response_time: float,
        confidence: float,
        error: Optional[str] = None,
    ) -> None:
        """Record a new observation."""
        obs = Observation(
            output=output,
            response_time=response_time,
            confidence=confidence,
            error=error,
        )
        self.observations.append(obs)

        # Enforce window size
        if len(self.observations) > self.window_size:
            self.observations.pop(0)

    def record_canary_result(self, passed: bool) -> None:
        """Record result of a canary test."""
        self.canary_results.append(passed)

    def generate_peptide(self) -> Optional[MHCPeptide]:
        """Generate behavioral fingerprint from collected observations."""
        if len(self.observations) < self.min_observations:
            return None

        # Calculate statistics
        lengths = [len(obs.output) if obs.output else 0 for obs in self.observations]
        times = [obs.response_time for obs in self.observations]
        confidences = [obs.confidence for obs in self.observations]

        # Error tracking
        errors = [obs.error for obs in self.observations if obs.error]
        error_rate = len(errors) / len(self.observations)
        error_types = tuple(sorted(set(errors)))

        # Vocabulary hash (sorted unique words)
        vocabulary = set()
        for obs in self.observations:
            if obs.output:
                words = re.findall(r'\b\w+\b', obs.output.lower())
                vocabulary.update(words)
        vocab_str = ",".join(sorted(vocabulary))
        vocab_hash = hashlib.md5(vocab_str.encode()).hexdigest()[:12]

        # Structure hash (detect JSON, lists, etc.)
        structures = []
        for obs in self.observations:
            if obs.output:
                structures.append(self._detect_structure(obs.output))
        struct_str = ",".join(sorted(set(structures)))
        struct_hash = hashlib.md5(struct_str.encode()).hexdigest()[:12]

        # Canary accuracy
        canary_accuracy = None
        if self.canary_results:
            canary_accuracy = sum(self.canary_results) / len(self.canary_results)

        return MHCPeptide(
            agent_id=self.agent_id,
            timestamp=datetime.utcnow(),
            output_length_mean=statistics.mean(lengths),
            output_length_std=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            response_time_mean=statistics.mean(times),
            response_time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
            vocabulary_hash=vocab_hash,
            structure_hash=struct_hash,
            confidence_mean=statistics.mean(confidences),
            confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            error_rate=error_rate,
            error_types=error_types,
            canary_accuracy=canary_accuracy,
        )

    def _detect_structure(self, output: str) -> str:
        """Detect output structure type."""
        output = output.strip()

        # Try JSON
        if output.startswith('{') or output.startswith('['):
            try:
                json.loads(output)
                return "json"
            except:
                pass

        # Check for common patterns
        if re.match(r'^\d+\.\s', output):
            return "numbered_list"
        if re.match(r'^[-*]\s', output):
            return "bullet_list"
        if re.match(r'^#', output):
            return "markdown"

        return "plain"

    def clear(self) -> None:
        """Clear all observations and canary results."""
        self.observations.clear()
        self.canary_results.clear()
