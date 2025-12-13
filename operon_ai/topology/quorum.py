"""
Quorum Sensing: Consensus-Based Decision Making
================================================

Biological Analogy:
- Autoinducers: Signal molecules secreted by bacteria
- Quorum threshold: Minimum population density for collective behavior
- Biofilm formation: Coordinated response when quorum is reached
- Competence: Ability to uptake genetic material when quorum reached
- Bioluminescence: Synchronized light production in some bacteria

The QuorumSensing topology enables multi-agent consensus decisions
with configurable voting strategies, weighted influence, and
confidence-based aggregation.

Note: quorum improves reliability when votes are not perfectly correlated.
If agents share the same prompts/context/tools, a "confidently wrong" quorum
is still possible; inject diversity and tool-grounded checks where possible.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
from datetime import datetime
import threading
import math

from ..core.agent import BioAgent
from ..core.types import Signal, ActionProtein
from ..state.metabolism import ATP_Store


class VotingStrategy(Enum):
    """Strategies for aggregating votes."""
    MAJORITY = "majority"           # >50% required
    SUPERMAJORITY = "supermajority" # >66% required
    UNANIMOUS = "unanimous"         # 100% required
    WEIGHTED = "weighted"           # Weight-adjusted majority
    CONFIDENCE = "confidence"       # Confidence-weighted average
    BAYESIAN = "bayesian"          # Bayesian belief aggregation
    THRESHOLD = "threshold"         # Fixed threshold count


class VoteType(Enum):
    """Types of votes an agent can cast."""
    PERMIT = "permit"
    BLOCK = "block"
    ABSTAIN = "abstain"
    DEFER = "defer"  # Defer to higher-weight agents


@dataclass
class Vote:
    """A single vote from an agent."""
    agent_id: str
    vote_type: VoteType
    confidence: float = 1.0  # 0-1 confidence in the vote
    weight: float = 1.0      # Agent's voting weight
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def effective_weight(self) -> float:
        """Weight adjusted by confidence."""
        return self.weight * self.confidence


@dataclass
class QuorumResult:
    """Result of a quorum sensing vote."""
    reached: bool
    decision: VoteType
    total_votes: int
    permit_votes: int
    block_votes: int
    abstain_votes: int
    weighted_score: float
    confidence_score: float
    threshold_used: float
    strategy: VotingStrategy
    votes: list[Vote] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class AgentProfile:
    """Profile for a voting agent."""
    agent: BioAgent
    weight: float = 1.0
    reliability_score: float = 1.0  # Track historical accuracy
    votes_cast: int = 0
    correct_votes: int = 0


class QuorumSensing:
    """
    Consensus-Based Decision Topology.

    Implements multi-agent voting with various strategies inspired
    by bacterial quorum sensing mechanisms.

    Features:

    1. Voting Strategies
       - MAJORITY: Simple >50% majority
       - SUPERMAJORITY: >66% required (2/3)
       - UNANIMOUS: All must agree
       - WEIGHTED: Weight-adjusted voting
       - CONFIDENCE: Confidence-weighted aggregation
       - BAYESIAN: Bayesian belief update
       - THRESHOLD: Fixed count threshold

    2. Weighted Voting
       - Agents have configurable weights
       - Reliability tracking adjusts weight
       - Dynamic weight adjustment

    3. Confidence Aggregation
       - Agents report confidence scores
       - Low-confidence votes weighted less
       - Uncertainty propagation

    4. Reliability Tracking
       - Track agent accuracy over time
       - Adjust weights based on performance
       - Identify unreliable agents

    5. Dynamic Quorum
       - Adjust threshold based on context
       - Emergency mode for critical decisions
       - Adaptive quorum for availability

    Example:
        >>> budget = ATP_Store(budget=500)
        >>> quorum = QuorumSensing(n_agents=5, budget=budget)
        >>> quorum.set_strategy(VotingStrategy.WEIGHTED)
        >>> result = quorum.run_vote("Should we proceed with deployment?")
        >>> if result.reached:
        ...     print("Consensus: Proceed")
    """

    # Default thresholds
    MAJORITY_THRESHOLD = 0.5
    SUPERMAJORITY_THRESHOLD = 0.666
    CONFIDENCE_MIN = 0.3  # Minimum confidence to count vote

    def __init__(
        self,
        n_agents: int,
        budget: ATP_Store,
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        threshold: float | None = None,
        min_voters: int = 1,
        timeout_seconds: float = 30.0,
        enable_reliability_tracking: bool = True,
        on_quorum_reached: Callable[[QuorumResult], None] | None = None,
        on_quorum_failed: Callable[[QuorumResult], None] | None = None,
        silent: bool = False,
    ):
        """
        Initialize QuorumSensing.

        Args:
            n_agents: Number of voting agents
            budget: Shared ATP budget
            strategy: Voting strategy to use
            threshold: Custom threshold (overrides strategy default)
            min_voters: Minimum voters required for valid quorum
            timeout_seconds: Timeout for vote collection
            enable_reliability_tracking: Track agent reliability
            on_quorum_reached: Callback when quorum is reached
            on_quorum_failed: Callback when quorum fails
            silent: Suppress console output
        """
        self.budget = budget
        self.strategy = strategy
        self.custom_threshold = threshold
        self.min_voters = min_voters
        self.timeout_seconds = timeout_seconds
        self.enable_reliability_tracking = enable_reliability_tracking
        self.on_quorum_reached = on_quorum_reached
        self.on_quorum_failed = on_quorum_failed
        self.silent = silent

        # Create agents with profiles
        self.colony: list[AgentProfile] = []
        for i in range(n_agents):
            agent = BioAgent(f"Bacterium_{i}", "Voter", budget)
            profile = AgentProfile(agent=agent, weight=1.0)
            self.colony.append(profile)

        # Statistics
        self._total_votes = 0
        self._quorums_reached = 0
        self._quorums_failed = 0
        self._vote_history: list[QuorumResult] = []
        self._lock = threading.Lock()

    def add_agent(self, name: str, weight: float = 1.0) -> AgentProfile:
        """Add a new voting agent."""
        agent = BioAgent(name, "Voter", self.budget)
        profile = AgentProfile(agent=agent, weight=weight)
        self.colony.append(profile)
        if not self.silent:
            print(f"🦠 [Quorum] Added agent: {name} (weight: {weight})")
        return profile

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the colony."""
        for i, profile in enumerate(self.colony):
            if profile.agent.name == name:
                self.colony.pop(i)
                if not self.silent:
                    print(f"🦠 [Quorum] Removed agent: {name}")
                return True
        return False

    def set_agent_weight(self, name: str, weight: float) -> bool:
        """Set an agent's voting weight."""
        for profile in self.colony:
            if profile.agent.name == name:
                profile.weight = weight
                return True
        return False

    def set_strategy(self, strategy: VotingStrategy, threshold: float | None = None):
        """Change the voting strategy."""
        self.strategy = strategy
        self.custom_threshold = threshold
        if not self.silent:
            print(f"📊 [Quorum] Strategy: {strategy.value}")

    def run_vote(self, prompt: str, context: dict[str, Any] | None = None) -> QuorumResult:
        """
        Run a vote across all agents.

        Args:
            prompt: The question/proposal to vote on
            context: Optional context for agents

        Returns:
            QuorumResult with decision and details
        """
        import time
        start_time = time.time()

        if not self.silent:
            print(f"\n📢 [Quorum] Initiating Vote on: '{prompt}'")

        signal = Signal(content=prompt)
        votes: list[Vote] = []

        # Collect votes from all agents
        for profile in self.colony:
            try:
                protein = profile.agent.express(signal)
                vote = self._protein_to_vote(protein, profile)
                votes.append(vote)
                profile.votes_cast += 1
            except Exception as e:
                if not self.silent:
                    print(f"⚠️ [Quorum] Agent {profile.agent.name} failed: {e}")
                # Record abstain for failed agents
                votes.append(Vote(
                    agent_id=profile.agent.name,
                    vote_type=VoteType.ABSTAIN,
                    confidence=0.0,
                    weight=profile.weight,
                    reasoning=f"Error: {e}"
                ))

        # Aggregate votes based on strategy
        result = self._aggregate_votes(votes)
        result.processing_time_ms = (time.time() - start_time) * 1000

        # Record statistics
        self._total_votes += len(votes)
        self._vote_history.append(result)
        if len(self._vote_history) > 1000:
            self._vote_history = self._vote_history[-1000:]

        # Callbacks and logging
        if result.reached:
            self._quorums_reached += 1
            if self.on_quorum_reached:
                self.on_quorum_reached(result)
        else:
            self._quorums_failed += 1
            if self.on_quorum_failed:
                self.on_quorum_failed(result)

        # Console output
        if not self.silent:
            self._print_result(result)

        return result

    def _protein_to_vote(self, protein: ActionProtein, profile: AgentProfile) -> Vote:
        """Convert agent output to a vote."""
        # Determine vote type from action
        if protein.action_type in ("PERMIT", "EXECUTE"):
            vote_type = VoteType.PERMIT
        elif protein.action_type == "BLOCK":
            vote_type = VoteType.BLOCK
        elif protein.action_type == "DEFER":
            vote_type = VoteType.DEFER
        else:
            vote_type = VoteType.ABSTAIN

        # Extract confidence from payload if present
        confidence = 1.0
        if isinstance(protein.payload, dict) and "confidence" in protein.payload:
            confidence = float(protein.payload["confidence"])

        return Vote(
            agent_id=profile.agent.name,
            vote_type=vote_type,
            confidence=confidence,
            weight=profile.weight * profile.reliability_score,
            reasoning=str(protein.payload) if protein.payload else ""
        )

    def _aggregate_votes(self, votes: list[Vote]) -> QuorumResult:
        """Aggregate votes using the configured strategy."""
        # Count votes by type
        permit_votes = [v for v in votes if v.vote_type == VoteType.PERMIT]
        block_votes = [v for v in votes if v.vote_type == VoteType.BLOCK]
        abstain_votes = [v for v in votes if v.vote_type == VoteType.ABSTAIN]
        defer_votes = [v for v in votes if v.vote_type == VoteType.DEFER]

        total_votes = len(votes) - len(abstain_votes) - len(defer_votes)

        if total_votes < self.min_voters:
            return QuorumResult(
                reached=False,
                decision=VoteType.ABSTAIN,
                total_votes=len(votes),
                permit_votes=len(permit_votes),
                block_votes=len(block_votes),
                abstain_votes=len(abstain_votes),
                weighted_score=0.0,
                confidence_score=0.0,
                threshold_used=0.0,
                strategy=self.strategy,
                votes=votes
            )

        # Calculate scores based on strategy
        if self.strategy == VotingStrategy.MAJORITY:
            return self._simple_majority(votes, permit_votes, block_votes, abstain_votes)

        elif self.strategy == VotingStrategy.SUPERMAJORITY:
            return self._supermajority(votes, permit_votes, block_votes, abstain_votes)

        elif self.strategy == VotingStrategy.UNANIMOUS:
            return self._unanimous(votes, permit_votes, block_votes, abstain_votes)

        elif self.strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(votes, permit_votes, block_votes, abstain_votes)

        elif self.strategy == VotingStrategy.CONFIDENCE:
            return self._confidence_vote(votes, permit_votes, block_votes, abstain_votes)

        elif self.strategy == VotingStrategy.BAYESIAN:
            return self._bayesian_vote(votes, permit_votes, block_votes, abstain_votes)

        elif self.strategy == VotingStrategy.THRESHOLD:
            return self._threshold_vote(votes, permit_votes, block_votes, abstain_votes)

        # Fallback to majority
        return self._simple_majority(votes, permit_votes, block_votes, abstain_votes)

    def _simple_majority(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Simple majority: >50% permits."""
        threshold = self.custom_threshold or self.MAJORITY_THRESHOLD
        active_votes = len(permit_votes) + len(block_votes)

        if active_votes == 0:
            ratio = 0.0
        else:
            ratio = len(permit_votes) / active_votes

        reached = ratio > threshold
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=ratio,
            confidence_score=1.0,
            threshold_used=threshold,
            strategy=self.strategy,
            votes=votes
        )

    def _supermajority(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Supermajority: >66% permits."""
        threshold = self.custom_threshold or self.SUPERMAJORITY_THRESHOLD
        active_votes = len(permit_votes) + len(block_votes)

        if active_votes == 0:
            ratio = 0.0
        else:
            ratio = len(permit_votes) / active_votes

        reached = ratio > threshold
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=ratio,
            confidence_score=1.0,
            threshold_used=threshold,
            strategy=self.strategy,
            votes=votes
        )

    def _unanimous(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Unanimous: All must permit."""
        active_votes = len(permit_votes) + len(block_votes)
        reached = len(block_votes) == 0 and len(permit_votes) > 0
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        ratio = 1.0 if reached else (len(permit_votes) / max(1, active_votes))

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=ratio,
            confidence_score=1.0,
            threshold_used=1.0,
            strategy=self.strategy,
            votes=votes
        )

    def _weighted_vote(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Weighted voting: Weight-adjusted majority."""
        threshold = self.custom_threshold or self.MAJORITY_THRESHOLD

        permit_weight = sum(v.effective_weight for v in permit_votes)
        block_weight = sum(v.effective_weight for v in block_votes)
        total_weight = permit_weight + block_weight

        if total_weight == 0:
            ratio = 0.0
        else:
            ratio = permit_weight / total_weight

        reached = ratio > threshold
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=ratio,
            confidence_score=1.0,
            threshold_used=threshold,
            strategy=self.strategy,
            votes=votes
        )

    def _confidence_vote(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Confidence-weighted: Only count votes above confidence threshold."""
        threshold = self.custom_threshold or self.MAJORITY_THRESHOLD

        # Filter by minimum confidence
        confident_permits = [v for v in permit_votes if v.confidence >= self.CONFIDENCE_MIN]
        confident_blocks = [v for v in block_votes if v.confidence >= self.CONFIDENCE_MIN]

        permit_score = sum(v.effective_weight for v in confident_permits)
        block_score = sum(v.effective_weight for v in confident_blocks)
        total_score = permit_score + block_score

        if total_score == 0:
            ratio = 0.0
            avg_confidence = 0.0
        else:
            ratio = permit_score / total_score
            all_confident = confident_permits + confident_blocks
            avg_confidence = sum(v.confidence for v in all_confident) / len(all_confident)

        reached = ratio > threshold
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=ratio,
            confidence_score=avg_confidence,
            threshold_used=threshold,
            strategy=self.strategy,
            votes=votes
        )

    def _bayesian_vote(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Bayesian belief aggregation."""
        threshold = self.custom_threshold or self.MAJORITY_THRESHOLD

        # Start with uniform prior
        prior_permit = 0.5
        prior_block = 0.5

        # Update belief based on each vote
        for vote in permit_votes:
            # Higher confidence = more influence
            likelihood = 0.5 + (vote.confidence * 0.4)  # 0.5-0.9
            prior_permit = self._bayesian_update(prior_permit, likelihood, vote.weight)

        for vote in block_votes:
            likelihood = 0.5 + (vote.confidence * 0.4)
            prior_block = self._bayesian_update(prior_block, likelihood, vote.weight)

        # Normalize
        total = prior_permit + prior_block
        if total > 0:
            posterior_permit = prior_permit / total
        else:
            posterior_permit = 0.5

        reached = posterior_permit > threshold
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=posterior_permit,
            confidence_score=abs(posterior_permit - 0.5) * 2,  # How certain
            threshold_used=threshold,
            strategy=self.strategy,
            votes=votes
        )

    def _bayesian_update(self, prior: float, likelihood: float, weight: float) -> float:
        """Apply Bayesian update with weighted evidence."""
        # Weighted likelihood based on agent weight
        adjusted_likelihood = 0.5 + (likelihood - 0.5) * weight

        # Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
        # Simplified: just multiply prior by likelihood
        return prior * adjusted_likelihood

    def _threshold_vote(
        self,
        votes: list[Vote],
        permit_votes: list[Vote],
        block_votes: list[Vote],
        abstain_votes: list[Vote]
    ) -> QuorumResult:
        """Fixed threshold count (e.g., need exactly N permits)."""
        threshold = int(self.custom_threshold or len(self.colony) // 2 + 1)

        reached = len(permit_votes) >= threshold
        decision = VoteType.PERMIT if reached else VoteType.BLOCK

        active_votes = len(permit_votes) + len(block_votes)
        ratio = len(permit_votes) / max(1, active_votes)

        return QuorumResult(
            reached=reached,
            decision=decision,
            total_votes=len(votes),
            permit_votes=len(permit_votes),
            block_votes=len(block_votes),
            abstain_votes=len(abstain_votes),
            weighted_score=ratio,
            confidence_score=1.0,
            threshold_used=threshold / len(self.colony),
            strategy=self.strategy,
            votes=votes
        )

    def update_reliability(self, agent_name: str, was_correct: bool):
        """Update an agent's reliability score based on outcome."""
        if not self.enable_reliability_tracking:
            return

        for profile in self.colony:
            if profile.agent.name == agent_name:
                if was_correct:
                    profile.correct_votes += 1
                # Recalculate reliability
                if profile.votes_cast > 0:
                    profile.reliability_score = profile.correct_votes / profile.votes_cast
                break

    def update_all_reliability(self, correct_decision: VoteType):
        """Update reliability for all agents based on the correct decision."""
        if not self.enable_reliability_tracking or not self._vote_history:
            return

        last_result = self._vote_history[-1]
        for vote in last_result.votes:
            was_correct = vote.vote_type == correct_decision
            self.update_reliability(vote.agent_id, was_correct)

    def _print_result(self, result: QuorumResult):
        """Print vote result to console."""
        print(f"📊 [Quorum] Results ({result.strategy.value}):")
        print(f"   Permits: {result.permit_votes}, Blocks: {result.block_votes}, "
              f"Abstains: {result.abstain_votes}")
        print(f"   Weighted Score: {result.weighted_score:.2%}")
        print(f"   Confidence: {result.confidence_score:.2%}")
        print(f"   Threshold: {result.threshold_used:.2%}")

        if result.reached:
            print(f"✅ QUORUM REACHED: {result.decision.value.upper()}")
        else:
            print(f"❌ QUORUM FAILED: {result.decision.value.upper()}")

    def get_statistics(self) -> dict:
        """Get quorum sensing statistics."""
        agent_stats = []
        for profile in self.colony:
            agent_stats.append({
                "name": profile.agent.name,
                "weight": profile.weight,
                "reliability": profile.reliability_score,
                "votes_cast": profile.votes_cast,
                "correct_votes": profile.correct_votes,
            })

        return {
            "n_agents": len(self.colony),
            "strategy": self.strategy.value,
            "total_votes": self._total_votes,
            "quorums_reached": self._quorums_reached,
            "quorums_failed": self._quorums_failed,
            "success_rate": self._quorums_reached / max(1, self._quorums_reached + self._quorums_failed),
            "agent_stats": agent_stats,
        }

    def get_vote_history(self, limit: int = 100) -> list[QuorumResult]:
        """Get recent vote history."""
        return self._vote_history[-limit:]

    def get_agent_rankings(self) -> list[dict]:
        """Get agents ranked by reliability."""
        rankings = []
        for profile in self.colony:
            rankings.append({
                "name": profile.agent.name,
                "reliability": profile.reliability_score,
                "effective_weight": profile.weight * profile.reliability_score,
                "votes_cast": profile.votes_cast,
            })
        return sorted(rankings, key=lambda x: x["effective_weight"], reverse=True)


class EmergencyQuorum(QuorumSensing):
    """
    Emergency Quorum for time-critical decisions.

    Uses reduced thresholds and timeouts for urgent situations.
    """

    def __init__(
        self,
        n_agents: int,
        budget: ATP_Store,
        emergency_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(
            n_agents=n_agents,
            budget=budget,
            strategy=VotingStrategy.THRESHOLD,
            threshold=emergency_threshold,
            min_voters=1,
            timeout_seconds=5.0,
            **kwargs
        )
        if not self.silent:
            print("🚨 [Quorum] Emergency mode activated")

    def run_vote(self, prompt: str, context: dict[str, Any] | None = None) -> QuorumResult:
        """Run emergency vote with reduced requirements."""
        if not self.silent:
            print("🚨 [Quorum] EMERGENCY VOTE")
        return super().run_vote(prompt, context)
