#!/usr/bin/env python3
"""
Example 50: Morphogen Cascade with Quorum
==========================================

Demonstrates a multi-stage document/contract review where cascade stages
check morphogen signals and escalate to quorum vote when confidence is low.

Architecture:

```
Document Input
    |
[MorphogenGradient] <- initialized from document metadata
    |
[Cascade Stage 1: Extract Clauses]
    ├── Parse document
    └── Update COMPLEXITY morphogen
    |
[Cascade Stage 2: Risk Assessment]
    ├── IF confidence HIGH: single reviewer
    └── IF confidence LOW: QuorumSensing with 3 voters
    |
[Cascade Stage 3: Final Decision]
    ├── Read RISK morphogen
    ├── Determine approval threshold
    └── Render verdict
    |
[NegativeFeedbackLoop]
    └── Adjusts confidence based on voting outcomes
```

Key concepts:
- Morphogen gradients carry document metadata (complexity, risk) across stages
- QuorumSensing activates dynamically based on confidence level
- NegativeFeedbackLoop adjusts confidence thresholds over time
- Each cascade stage reads and writes morphogen signals

Prerequisites:
- Example 42 for Morphogen gradient concepts
- Example 06 for Quorum Sensing
- Example 02 for Negative Feedback Loop

Usage:
    python examples/50_morphogen_cascade_quorum.py
    python examples/50_morphogen_cascade_quorum.py --test
"""

import sys
from dataclasses import dataclass, field
from enum import Enum

from operon_ai import ATP_Store
from operon_ai.coordination.morphogen import (
    MorphogenGradient,
    MorphogenType,
)
from operon_ai.topology.quorum import (
    QuorumSensing,
    VotingStrategy,
)


# =============================================================================
# Data Structures
# =============================================================================


class RiskLevel(str, Enum):
    """Risk level for a document."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewVerdict(str, Enum):
    """Final review verdict."""
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"


@dataclass
class DocumentClause:
    """A clause extracted from a document."""
    id: int
    text: str
    risk_indicators: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class ReviewReport:
    """Complete review report."""
    document_id: str
    verdict: ReviewVerdict
    risk_level: RiskLevel
    confidence: float
    clauses_reviewed: int
    quorum_used: bool
    quorum_result: str | None = None
    risk_factors: list[str] = field(default_factory=list)
    recommendation: str = ""


# =============================================================================
# Clause Extractor (Stage 1)
# =============================================================================


class ClauseExtractor:
    """
    Extracts and analyzes clauses from documents.

    Updates the COMPLEXITY morphogen based on document structure.
    """

    # Risk indicator keywords
    RISK_KEYWORDS = {
        "indemnify": 0.3,
        "liability": 0.2,
        "penalty": 0.3,
        "termination": 0.2,
        "unlimited": 0.4,
        "irrevocable": 0.3,
        "waive": 0.3,
        "exclusive": 0.2,
        "non-compete": 0.3,
        "damages": 0.2,
        "confidential": 0.1,
        "warranty": 0.1,
    }

    def __init__(self, silent: bool = False):
        self.silent = silent

    def extract(
        self,
        document: str,
        gradient: MorphogenGradient,
    ) -> list[DocumentClause]:
        """
        Extract clauses from document and update gradient.
        """
        # Split document into clauses (simple: by paragraph)
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        clauses: list[DocumentClause] = []

        total_risk = 0.0

        for i, para in enumerate(paragraphs):
            # Find risk indicators
            indicators = []
            risk_score = 0.0
            for keyword, weight in self.RISK_KEYWORDS.items():
                if keyword.lower() in para.lower():
                    indicators.append(keyword)
                    risk_score += weight

            clause = DocumentClause(
                id=i + 1,
                text=para[:200],
                risk_indicators=indicators,
                risk_score=min(1.0, risk_score),
            )
            clauses.append(clause)
            total_risk += risk_score

        # Update morphogens
        num_clauses = len(clauses)
        complexity = min(1.0, num_clauses / 10.0)  # More clauses = more complex
        avg_risk = total_risk / max(1, num_clauses)

        gradient.set(MorphogenType.COMPLEXITY, complexity,
                     description=f"Document complexity ({num_clauses} clauses)")
        gradient.set(MorphogenType.RISK, min(1.0, avg_risk),
                     description=f"Average clause risk ({avg_risk:.2f})")

        if not self.silent:
            print(f"  [Stage 1] Extracted {num_clauses} clauses")
            print(f"  [Stage 1] Complexity: {complexity:.2f}, Avg risk: {avg_risk:.2f}")
            risky = [c for c in clauses if c.risk_indicators]
            if risky:
                print(f"  [Stage 1] Risky clauses: {len(risky)}")
                for c in risky[:3]:
                    print(f"    Clause {c.id}: {', '.join(c.risk_indicators)}")

        return clauses


# =============================================================================
# Risk Assessor (Stage 2)
# =============================================================================


class RiskAssessor:
    """
    Assesses risk from extracted clauses.

    When confidence is LOW, activates QuorumSensing with multiple reviewers.
    When confidence is HIGH, uses a single fast assessment.
    """

    def __init__(self, silent: bool = False):
        self.silent = silent

    def assess(
        self,
        clauses: list[DocumentClause],
        gradient: MorphogenGradient,
        budget: ATP_Store,
    ) -> tuple[RiskLevel, float, bool, str | None]:
        """
        Assess document risk.

        Returns:
            (risk_level, confidence, quorum_used, quorum_details)
        """
        confidence = gradient.get(MorphogenType.CONFIDENCE)
        complexity = gradient.get(MorphogenType.COMPLEXITY)

        # Calculate aggregate risk
        max_risk = max((c.risk_score for c in clauses), default=0.0)
        avg_risk = sum(c.risk_score for c in clauses) / max(1, len(clauses))
        risk_factors = []
        for c in clauses:
            risk_factors.extend(c.risk_indicators)

        if not self.silent:
            print(f"\n  [Stage 2] Confidence: {confidence:.2f}, Complexity: {complexity:.2f}")

        # Decide: single reviewer or quorum?
        need_quorum = confidence < 0.5 or (complexity > 0.6 and max_risk > 0.3)

        quorum_details = None

        if need_quorum and budget.consume(30, "quorum_review"):
            # Activate QuorumSensing
            if not self.silent:
                print("  [Stage 2] LOW confidence -> activating Quorum (3 reviewers)")

            quorum_result = self._run_quorum_review(clauses, gradient)
            quorum_details = quorum_result

            # Quorum adjusts confidence
            if "permit" in quorum_result.lower():
                gradient.set(MorphogenType.CONFIDENCE,
                             min(1.0, confidence + 0.2))
            else:
                gradient.set(MorphogenType.CONFIDENCE,
                             max(0.0, confidence - 0.1))
        elif not need_quorum:
            if not self.silent:
                print("  [Stage 2] HIGH confidence -> single reviewer")
            budget.consume(10, "single_review")
        else:
            if not self.silent:
                print("  [Stage 2] Insufficient budget for quorum, using single reviewer")
            budget.consume(10, "single_review_fallback")

        # Determine risk level
        if max_risk >= 0.6 or avg_risk >= 0.4:
            risk_level = RiskLevel.CRITICAL
        elif max_risk >= 0.4 or avg_risk >= 0.25:
            risk_level = RiskLevel.HIGH
        elif max_risk >= 0.2 or avg_risk >= 0.1:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Update risk morphogen
        gradient.set(MorphogenType.RISK, max_risk,
                     description=f"Max clause risk: {max_risk:.2f}")

        if not self.silent:
            print(f"  [Stage 2] Risk level: {risk_level.value} "
                  f"(max={max_risk:.2f}, avg={avg_risk:.2f})")

        return risk_level, confidence, need_quorum, quorum_details

    def _run_quorum_review(
        self,
        clauses: list[DocumentClause],
        gradient: MorphogenGradient,
    ) -> str:
        """Run quorum review with 3 simulated reviewers."""
        quorum_budget = ATP_Store(budget=300, silent=True)

        quorum = QuorumSensing(
            n_agents=3,
            budget=quorum_budget,
            strategy=VotingStrategy.MAJORITY,
            silent=True,
        )

        # Build voting question from risk analysis
        risky_clauses = [c for c in clauses if c.risk_score > 0.2]
        risk_summary = ", ".join(
            f"clause {c.id} ({c.risk_score:.1f})"
            for c in risky_clauses[:5]
        )

        question = f"Should this contract be approved? Risky clauses: {risk_summary}"
        result = quorum.run_vote(question)

        if not self.silent:
            print(f"  [Quorum] Decision: {result.decision.value} "
                  f"(reached={result.reached}, score={result.weighted_score:.2f})")
            for vote in result.votes:
                print(f"    {vote.agent_id}: {vote.vote_type.value}")

        return f"Quorum: {result.decision.value} (score={result.weighted_score:.2f})"


# =============================================================================
# Decision Maker (Stage 3)
# =============================================================================


class DecisionMaker:
    """
    Makes final verdict based on risk assessment and morphogen state.

    Uses NegativeFeedbackLoop-style threshold adjustment:
    lower confidence -> higher approval threshold (more conservative).
    """

    def __init__(self, silent: bool = False):
        self.silent = silent
        # Track threshold adjustments for feedback
        self._threshold_history: list[float] = []

    def decide(
        self,
        risk_level: RiskLevel,
        clauses: list[DocumentClause],
        gradient: MorphogenGradient,
        quorum_used: bool,
        quorum_result: str | None,
    ) -> ReviewReport:
        """Make final review decision."""
        confidence = gradient.get(MorphogenType.CONFIDENCE)
        risk = gradient.get(MorphogenType.RISK)

        # Negative feedback: adjust approval threshold based on confidence
        # Low confidence -> require lower risk for approval (more conservative)
        base_threshold = 0.4  # Default: approve if risk < 0.4
        adjusted_threshold = base_threshold * (0.5 + confidence * 0.5)
        # When confidence is low (0.0), threshold = 0.2 (very strict)
        # When confidence is high (1.0), threshold = 0.4 (normal)
        self._threshold_history.append(adjusted_threshold)

        if not self.silent:
            print(f"\n  [Stage 3] Risk: {risk:.2f}, "
                  f"Confidence: {confidence:.2f}, "
                  f"Threshold: {adjusted_threshold:.2f}")

        # Determine verdict
        risk_factors = []
        for c in clauses:
            risk_factors.extend(c.risk_indicators)
        risk_factors = list(set(risk_factors))

        if risk <= adjusted_threshold and risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
            verdict = ReviewVerdict.APPROVED
            recommendation = "Document meets review standards."
        elif risk <= adjusted_threshold * 1.5:
            verdict = ReviewVerdict.CONDITIONAL
            recommendation = f"Conditional approval. Address: {', '.join(risk_factors[:3])}"
        else:
            verdict = ReviewVerdict.REJECTED
            recommendation = f"Rejected due to: {', '.join(risk_factors[:5])}"

        if not self.silent:
            print(f"  [Stage 3] Verdict: {verdict.value}")
            print(f"  [Stage 3] Recommendation: {recommendation}")

        return ReviewReport(
            document_id="doc",
            verdict=verdict,
            risk_level=risk_level,
            confidence=confidence,
            clauses_reviewed=len(clauses),
            quorum_used=quorum_used,
            quorum_result=quorum_result,
            risk_factors=risk_factors,
            recommendation=recommendation,
        )


# =============================================================================
# Compliance Review Pipeline
# =============================================================================


class ComplianceReviewPipeline:
    """
    Full compliance review pipeline with morphogen coordination.

    Orchestrates:
    1. Clause extraction + complexity morphogen
    2. Risk assessment + optional quorum
    3. Final decision with feedback-adjusted thresholds
    """

    def __init__(self, silent: bool = False):
        self.silent = silent
        self.gradient = MorphogenGradient()
        self.budget = ATP_Store(budget=500, silent=True)
        self.extractor = ClauseExtractor(silent=silent)
        self.assessor = RiskAssessor(silent=silent)
        self.decider = DecisionMaker(silent=silent)

        # Initialize confidence
        self.gradient.set(MorphogenType.CONFIDENCE, 0.5)

    def review(self, document: str, document_id: str = "doc") -> ReviewReport:
        """Run full review pipeline."""
        if not self.silent:
            print(f"\n{'='*60}")
            print(f"Reviewing document: {document_id}")
            print(f"{'='*60}")

        # Stage 1: Extract clauses
        clauses = self.extractor.extract(document, self.gradient)

        # Stage 2: Assess risk
        risk_level, confidence, quorum_used, quorum_result = self.assessor.assess(
            clauses, self.gradient, self.budget,
        )

        # Stage 3: Final decision
        report = self.decider.decide(
            risk_level, clauses, self.gradient,
            quorum_used, quorum_result,
        )
        report.document_id = document_id

        # Feedback: adjust confidence for next review based on outcome
        if report.verdict == ReviewVerdict.APPROVED:
            # Successful review -> boost confidence slightly
            current = self.gradient.get(MorphogenType.CONFIDENCE)
            self.gradient.set(MorphogenType.CONFIDENCE, min(1.0, current + 0.05))
        elif report.verdict == ReviewVerdict.REJECTED:
            # Rejection -> lower confidence (be more careful next time)
            current = self.gradient.get(MorphogenType.CONFIDENCE)
            self.gradient.set(MorphogenType.CONFIDENCE, max(0.0, current - 0.1))

        return report


# =============================================================================
# Demo Documents
# =============================================================================


SIMPLE_CONTRACT = """
Agreement for Widget Supply

This agreement is between Acme Corp and Widget Co for the supply
of standard widgets.

The supplier agrees to deliver 1000 widgets per month at
$5.00 per unit.

Payment terms are net 30 days from invoice date.

Both parties agree to maintain confidential any proprietary
information shared during this engagement.
"""

AMBIGUOUS_CONTRACT = """
Strategic Partnership Agreement

This partnership agreement covers joint development of
the new product line.

Either party may terminate this agreement with 30 days notice.
The termination clause includes a penalty of 2x monthly revenue.

All intellectual property developed jointly shall be shared
equally, with exclusive licensing rights to each party in
their respective territories.

Both parties agree to indemnify each other against claims
arising from the use of shared technology.

A non-compete clause prevents either party from developing
competing products for 24 months after termination.
"""

DANGEROUS_CONTRACT = """
Enterprise Licensing Agreement

The licensee accepts unlimited liability for any damages arising
from use of the software, including consequential damages.

The licensee irrevocably waives all warranty rights and agrees
to indemnify the licensor against any claims.

Termination may occur at licensor's sole discretion with
a penalty equal to the remaining contract value.

The licensee agrees to exclusive use of licensor's platform
and a non-compete preventing use of alternative solutions.

All disputes shall be resolved in licensor's jurisdiction.
The licensee waives right to jury trial.
"""


# =============================================================================
# Demos
# =============================================================================


def demo_simple_contract():
    """Demo: Simple contract with low risk."""
    print("=" * 60)
    print("Demo 1: Simple Contract (LOW risk)")
    print("=" * 60)

    pipeline = ComplianceReviewPipeline(silent=False)
    report = pipeline.review(SIMPLE_CONTRACT, "SIMPLE-001")

    print(f"\n  Verdict: {report.verdict.value}")
    print(f"  Risk: {report.risk_level.value}")
    print(f"  Confidence: {report.confidence:.2f}")
    print(f"  Quorum used: {report.quorum_used}")
    return report


def demo_ambiguous_contract():
    """Demo: Ambiguous contract triggers quorum."""
    print("\n" + "=" * 60)
    print("Demo 2: Ambiguous Contract (triggers quorum)")
    print("=" * 60)

    pipeline = ComplianceReviewPipeline(silent=False)
    # Lower confidence to force quorum
    pipeline.gradient.set(MorphogenType.CONFIDENCE, 0.3)
    report = pipeline.review(AMBIGUOUS_CONTRACT, "AMBIG-002")

    print(f"\n  Verdict: {report.verdict.value}")
    print(f"  Risk: {report.risk_level.value}")
    print(f"  Confidence: {report.confidence:.2f}")
    print(f"  Quorum used: {report.quorum_used}")
    if report.quorum_result:
        print(f"  Quorum result: {report.quorum_result}")
    print(f"  Risk factors: {', '.join(report.risk_factors[:5])}")
    return report


def demo_dangerous_contract():
    """Demo: Dangerous contract gets rejected."""
    print("\n" + "=" * 60)
    print("Demo 3: Dangerous Contract (HIGH risk, rejected)")
    print("=" * 60)

    pipeline = ComplianceReviewPipeline(silent=False)
    report = pipeline.review(DANGEROUS_CONTRACT, "DANGER-003")

    print(f"\n  Verdict: {report.verdict.value}")
    print(f"  Risk: {report.risk_level.value}")
    print(f"  Confidence: {report.confidence:.2f}")
    print(f"  Quorum used: {report.quorum_used}")
    print(f"  Risk factors: {', '.join(report.risk_factors[:5])}")
    print(f"  Recommendation: {report.recommendation}")
    return report


# =============================================================================
# Smoke Test
# =============================================================================


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    # Test 1: Simple contract approved
    pipeline = ComplianceReviewPipeline(silent=True)
    report = pipeline.review(SIMPLE_CONTRACT, "test-1")
    assert report.verdict == ReviewVerdict.APPROVED, (
        f"Simple contract should be approved, got {report.verdict}"
    )
    assert report.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)
    print("  Test 1: Simple contract approved - PASSED")

    # Test 2: Dangerous contract rejected or conditional
    pipeline2 = ComplianceReviewPipeline(silent=True)
    report2 = pipeline2.review(DANGEROUS_CONTRACT, "test-2")
    assert report2.verdict in (ReviewVerdict.REJECTED, ReviewVerdict.CONDITIONAL), (
        f"Dangerous contract should not be approved, got {report2.verdict}"
    )
    assert report2.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    print("  Test 2: Dangerous contract flagged - PASSED")

    # Test 3: Clause extraction
    gradient = MorphogenGradient()
    extractor = ClauseExtractor(silent=True)
    clauses = extractor.extract(AMBIGUOUS_CONTRACT, gradient)
    assert len(clauses) >= 3, f"Should extract multiple clauses, got {len(clauses)}"
    risky = [c for c in clauses if c.risk_indicators]
    assert len(risky) >= 1, "Should find risky clauses"
    print("  Test 3: Clause extraction - PASSED")

    # Test 4: Morphogen updates after extraction
    complexity = gradient.get(MorphogenType.COMPLEXITY)
    assert complexity > 0, "Complexity should be set"
    risk = gradient.get(MorphogenType.RISK)
    assert risk > 0, "Risk should be set"
    print("  Test 4: Morphogen updates - PASSED")

    # Test 5: Quorum triggered on low confidence
    pipeline3 = ComplianceReviewPipeline(silent=True)
    pipeline3.gradient.set(MorphogenType.CONFIDENCE, 0.2)  # Very low
    report3 = pipeline3.review(AMBIGUOUS_CONTRACT, "test-3")
    assert report3.quorum_used, "Should trigger quorum on low confidence"
    print("  Test 5: Quorum activation - PASSED")

    # Test 6: Confidence feedback loop
    pipeline4 = ComplianceReviewPipeline(silent=True)
    initial_confidence = pipeline4.gradient.get(MorphogenType.CONFIDENCE)
    pipeline4.review(SIMPLE_CONTRACT, "test-4a")  # Approved -> boost
    after_approval = pipeline4.gradient.get(MorphogenType.CONFIDENCE)
    assert after_approval >= initial_confidence, "Confidence should not drop on approval"
    print("  Test 6: Confidence feedback - PASSED")

    print("\nSmoke tests passed!")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Example 50: Morphogen Cascade with Quorum")
    print("MorphogenGradient + Cascade + QuorumSensing + NegativeFeedbackLoop")
    print("=" * 60)

    demo_simple_contract()
    demo_ambiguous_contract()
    demo_dangerous_contract()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The Morphogen Cascade with Quorum demonstrates dynamic orchestration:

1. MorphogenGradient carries document metadata across stages:
   - COMPLEXITY: How many clauses, structural complexity
   - RISK: Maximum and average clause risk scores
   - CONFIDENCE: System's confidence in its assessment

2. Cascade Pipeline: Extract -> Assess -> Decide
   Each stage reads and writes morphogen signals

3. QuorumSensing: Activated dynamically when confidence is LOW
   - 3 reviewers vote on ambiguous cases
   - Majority decision adjusts confidence

4. NegativeFeedbackLoop: Confidence adjusts over time
   - Approvals boost confidence (less cautious)
   - Rejections lower confidence (more cautious)
   - Lower confidence -> stricter approval threshold

Key insight: The morphogen gradient creates implicit coordination
between stages without explicit message passing. Each stage reads
the environment and adapts its behavior accordingly.
""")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_smoke_test()
    else:
        main()
