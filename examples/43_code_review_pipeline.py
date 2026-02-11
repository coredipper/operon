#!/usr/bin/env python3
"""
Example 43: Code Review Pipeline (CFFL + Multi-Organelle)
=========================================================

Demonstrates a practical code review automation system that uses:
- **CFFL (Coherent Feed-Forward Loop)**: Dual-approval gate requiring both
  security AND quality reviewers to approve before the review passes
- **Membrane**: Detects obvious malicious patterns in code before LLM analysis
- **ATP_Store**: Budgets large PRs to prevent runaway costs
- **Chaperone**: Validates structured review output against schema

This is a production-ready pattern for automated PR review in CI/CD pipelines.

Architecture:

```
PR Diff Input
    |
[Membrane: Detect malicious patterns]
    |
[ATP Budget Check: Reject if too large]
    |
+---------------------------+
|   CFFL Gate               |
|   +---------+ +---------+ |
|   |Security | |Quality  | |
|   |Reviewer | |Reviewer | |
|   +----+----+ +----+----+ |
|        |           |      |
|        +-----+-----+      |
|              v            |
|          [AND Gate]       |
+-------------+-------------+
              |
[Chaperone: Validate structured review]
              |
[Comment Generator: Format output]
```

Usage:
    python examples/43_code_review_pipeline.py              # Demo with mock data
    python examples/43_code_review_pipeline.py --test       # Smoke test
    python examples/43_code_review_pipeline.py --pr <diff>  # Review a diff file

Prerequisites:
- Example 01 for basic CFFL concepts
- Example 03 for Chaperone patterns
- Example 07 for Membrane patterns
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from operon_ai import ATP_Store, Chaperone
from operon_ai.core.types import Signal
from operon_ai.organelles.membrane import (
    Membrane,
    ThreatLevel,
    ThreatSignature,
)
from operon_ai.state.metabolism import MetabolicState


# =============================================================================
# Schema Definitions
# =============================================================================


class ReviewSeverity(str, Enum):
    """Severity level for review findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReviewCategory(str, Enum):
    """Category of review finding."""
    SECURITY = "security"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"


class ReviewFinding(BaseModel):
    """A single finding from the code review."""
    file: str
    line: int | None = None
    severity: ReviewSeverity
    category: ReviewCategory
    title: str
    description: str
    suggestion: str | None = None


class ReviewResult(BaseModel):
    """Complete code review result."""
    approved: bool
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    security_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    files_reviewed: int
    lines_reviewed: int
    review_time_ms: float = 0.0


# =============================================================================
# Code Pattern Definitions
# =============================================================================

# Security patterns to detect in code
SECURITY_PATTERNS = [
    # Dangerous system calls
    ThreatSignature(
        pattern=r"\bos\.system\s*\(",
        level=ThreatLevel.DANGEROUS,
        description="Direct shell command execution",
        is_regex=True,
    ),
    ThreatSignature(
        pattern=r"\bsubprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True",
        level=ThreatLevel.DANGEROUS,
        description="Shell injection risk",
        is_regex=True,
    ),
    ThreatSignature(
        pattern=r"\beval\s*\(",
        level=ThreatLevel.CRITICAL,
        description="Arbitrary code execution via eval()",
        is_regex=True,
    ),
    ThreatSignature(
        pattern=r"\bexec\s*\(",
        level=ThreatLevel.CRITICAL,
        description="Arbitrary code execution via exec()",
        is_regex=True,
    ),
    # SQL injection
    ThreatSignature(
        pattern=r"(?:execute|cursor\.execute)\s*\(\s*['\"].*%s",
        level=ThreatLevel.DANGEROUS,
        description="Potential SQL injection (string formatting)",
        is_regex=True,
    ),
    ThreatSignature(
        pattern=r"(?:execute|cursor\.execute)\s*\(\s*f['\"]",
        level=ThreatLevel.DANGEROUS,
        description="Potential SQL injection (f-string)",
        is_regex=True,
    ),
    # Hardcoded secrets
    ThreatSignature(
        pattern=r"(?:password|secret|api_key|token)\s*=\s*['\"][^'\"]{8,}['\"]",
        level=ThreatLevel.CRITICAL,
        description="Hardcoded secret detected",
        is_regex=True,
    ),
    # Pickle deserialization
    ThreatSignature(
        pattern=r"\bpickle\.loads?\s*\(",
        level=ThreatLevel.DANGEROUS,
        description="Unsafe deserialization (pickle)",
        is_regex=True,
    ),
    # Path traversal
    ThreatSignature(
        pattern=r"\.\./",
        level=ThreatLevel.SUSPICIOUS,
        description="Potential path traversal",
        is_regex=False,
    ),
]

# Quality patterns to check
QUALITY_PATTERNS = [
    # Missing error handling
    (r"except\s*:\s*(?:pass|\.\.\.)", "Bare except clause swallowing errors"),
    # Magic numbers
    (r"(?<!['\"])(?<!\w)\d{4,}(?!\w)(?!['\"])", "Magic number (consider named constant)"),
    # Long lines
    (r".{121,}", "Line exceeds 120 characters"),
    # TODO comments
    (r"#\s*TODO\b", "TODO comment found"),
    # Print statements (in production code)
    (r"\bprint\s*\(", "Print statement (use logging instead)"),
]


# =============================================================================
# Diff Parser
# =============================================================================

@dataclass
class DiffHunk:
    """A hunk of changes in a diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class ParsedDiff:
    """Parsed unified diff."""
    files: list[str]
    hunks: list[DiffHunk]
    additions: int
    deletions: int

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions


def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """
    Parse a unified diff into structured components.

    Handles standard unified diff format:
    --- a/file.py
    +++ b/file.py
    @@ -1,3 +1,4 @@
    """
    files: list[str] = []
    hunks: list[DiffHunk] = []
    additions = 0
    deletions = 0

    current_file = None
    current_hunk_lines: list[str] = []
    current_hunk_info = None

    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    for line in diff_text.split("\n"):
        # New file
        if line.startswith("+++ b/") or line.startswith("+++ "):
            current_file = line[6:] if line.startswith("+++ b/") else line[4:]
            if current_file and current_file not in files:
                files.append(current_file)

        # Hunk header
        elif line.startswith("@@"):
            # Save previous hunk
            if current_hunk_info and current_file:
                hunks.append(DiffHunk(
                    file_path=current_file,
                    old_start=current_hunk_info[0],
                    old_count=current_hunk_info[1],
                    new_start=current_hunk_info[2],
                    new_count=current_hunk_info[3],
                    lines=current_hunk_lines,
                ))

            match = hunk_pattern.match(line)
            if match:
                current_hunk_info = (
                    int(match.group(1)),
                    int(match.group(2) or 1),
                    int(match.group(3)),
                    int(match.group(4) or 1),
                )
                current_hunk_lines = []

        # Additions/deletions
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
            current_hunk_lines.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
            current_hunk_lines.append(line)
        elif line.startswith(" "):
            current_hunk_lines.append(line)

    # Save last hunk
    if current_hunk_info and current_file:
        hunks.append(DiffHunk(
            file_path=current_file,
            old_start=current_hunk_info[0],
            old_count=current_hunk_info[1],
            new_start=current_hunk_info[2],
            new_count=current_hunk_info[3],
            lines=current_hunk_lines,
        ))

    return ParsedDiff(
        files=files,
        hunks=hunks,
        additions=additions,
        deletions=deletions,
    )


# =============================================================================
# Reviewers
# =============================================================================

@dataclass
class ReviewerResult:
    """Result from a single reviewer."""
    approved: bool
    score: float
    findings: list[ReviewFinding]
    reviewer_name: str


class SecurityReviewer:
    """
    Reviews code for security vulnerabilities.

    Uses pattern matching (Membrane-like) for fast detection,
    with optional LLM analysis for deeper review.
    """

    def __init__(self, patterns: list[ThreatSignature] | None = None):
        self.patterns = patterns or SECURITY_PATTERNS
        self.membrane = Membrane(
            threshold=ThreatLevel.DANGEROUS,
            rate_limit=100,
            enable_adaptive=True,
        )
        # Add our patterns to membrane
        for pattern in self.patterns:
            self.membrane.add_signature(pattern)

    def review(self, diff: ParsedDiff) -> ReviewerResult:
        """Review diff for security issues."""
        findings: list[ReviewFinding] = []
        issues_found = 0

        for hunk in diff.hunks:
            # Only review added lines
            line_num = hunk.new_start
            for line in hunk.lines:
                if line.startswith("+"):
                    code = line[1:]  # Remove the + prefix

                    # Check against patterns
                    for pattern in self.patterns:
                        if pattern.matches(code):
                            findings.append(ReviewFinding(
                                file=hunk.file_path,
                                line=line_num,
                                severity=self._threat_to_severity(pattern.level),
                                category=ReviewCategory.SECURITY,
                                title=pattern.description,
                                description=f"Pattern detected: {pattern.pattern[:50]}...",
                                suggestion="Review and remediate this security concern",
                            ))
                            issues_found += 1

                    line_num += 1
                elif not line.startswith("-"):
                    line_num += 1

        # Calculate score (1.0 = no issues, 0.0 = critical issues)
        critical_count = sum(1 for f in findings if f.severity == ReviewSeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == ReviewSeverity.HIGH)

        score = 1.0
        score -= critical_count * 0.3  # Each critical is -30%
        score -= high_count * 0.15     # Each high is -15%
        score = max(0.0, score)

        # Approve if no critical/high issues
        approved = critical_count == 0 and high_count == 0

        return ReviewerResult(
            approved=approved,
            score=score,
            findings=findings,
            reviewer_name="SecurityReviewer",
        )

    def _threat_to_severity(self, threat: ThreatLevel) -> ReviewSeverity:
        """Convert threat level to review severity."""
        mapping = {
            ThreatLevel.CRITICAL: ReviewSeverity.CRITICAL,
            ThreatLevel.DANGEROUS: ReviewSeverity.HIGH,
            ThreatLevel.SUSPICIOUS: ReviewSeverity.MEDIUM,
            ThreatLevel.SAFE: ReviewSeverity.INFO,
        }
        return mapping.get(threat, ReviewSeverity.MEDIUM)


class QualityReviewer:
    """
    Reviews code for quality issues.

    Checks for code style, best practices, and maintainability.
    """

    def __init__(self, patterns: list[tuple[str, str]] | None = None):
        self.patterns = patterns or QUALITY_PATTERNS
        self._compiled = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in self.patterns
        ]

    def review(self, diff: ParsedDiff) -> ReviewerResult:
        """Review diff for quality issues."""
        findings: list[ReviewFinding] = []

        for hunk in diff.hunks:
            line_num = hunk.new_start
            for line in hunk.lines:
                if line.startswith("+"):
                    code = line[1:]

                    for pattern, description in self._compiled:
                        if pattern.search(code):
                            # Determine severity based on pattern
                            severity = ReviewSeverity.LOW
                            if "TODO" in description:
                                severity = ReviewSeverity.INFO
                            elif "error" in description.lower():
                                severity = ReviewSeverity.MEDIUM

                            findings.append(ReviewFinding(
                                file=hunk.file_path,
                                line=line_num,
                                severity=severity,
                                category=ReviewCategory.QUALITY,
                                title=description,
                                description=f"Found in: {code.strip()[:60]}...",
                            ))

                    line_num += 1
                elif not line.startswith("-"):
                    line_num += 1

        # Quality score based on findings
        medium_count = sum(1 for f in findings if f.severity == ReviewSeverity.MEDIUM)
        low_count = sum(1 for f in findings if f.severity == ReviewSeverity.LOW)

        score = 1.0
        score -= medium_count * 0.1  # Each medium is -10%
        score -= low_count * 0.03    # Each low is -3%
        score = max(0.0, score)

        # Quality reviewer is more lenient - approve unless score < 0.5
        approved = score >= 0.5

        return ReviewerResult(
            approved=approved,
            score=score,
            findings=findings,
            reviewer_name="QualityReviewer",
        )


# =============================================================================
# Code Review Pipeline
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the code review pipeline."""
    # ATP budget settings
    atp_per_file: int = 100
    max_files_without_approval: int = 20
    total_budget: int = 5000

    # Review thresholds
    min_security_score: float = 0.7
    min_quality_score: float = 0.5
    require_both_approvals: bool = True

    # Callbacks
    on_budget_warning: Callable[[str], None] | None = None
    on_review_complete: Callable[[ReviewResult], None] | None = None


class CodeReviewPipeline:
    """
    Production-ready code review pipeline using Operon patterns.

    Stages:
    1. Parse diff into structured format
    2. Check ATP budget (reject if too large without approval)
    3. Run Membrane pre-filter for obvious malicious patterns
    4. Run Security and Quality reviewers in parallel (CFFL)
    5. Validate output with Chaperone
    6. Generate structured review result
    """

    def __init__(self, config: PipelineConfig | None = None, silent: bool = False):
        self.config = config or PipelineConfig()
        self.silent = silent

        # Initialize Operon components
        self.budget = ATP_Store(
            budget=self.config.total_budget,
            on_state_change=self._handle_budget_state,
            silent=silent,
        )
        self.membrane = Membrane(
            threshold=ThreatLevel.CRITICAL,
            rate_limit=60,
        )
        self.chaperone = Chaperone(silent=silent)

        # Reviewers
        self.security_reviewer = SecurityReviewer()
        self.quality_reviewer = QualityReviewer()

        # Statistics
        self._reviews_count = 0
        self._approved_count = 0
        self._blocked_count = 0

    def _handle_budget_state(self, new_state: MetabolicState) -> None:
        """Handle budget state changes."""
        if new_state == MetabolicState.CONSERVING:
            if self.config.on_budget_warning:
                self.config.on_budget_warning("Budget is running low (>70% consumed)")
        elif new_state == MetabolicState.STARVING:
            if self.config.on_budget_warning:
                self.config.on_budget_warning("Budget is nearly exhausted (>90% consumed)")

    def review(self, diff_text: str, force_large: bool = False) -> ReviewResult:
        """
        Review a code diff.

        Args:
            diff_text: Unified diff text
            force_large: Allow large PRs without additional approval

        Returns:
            ReviewResult with findings and approval status
        """
        import time
        start_time = time.time()
        self._reviews_count += 1

        if not self.silent:
            print("\n" + "=" * 60)
            print("Code Review Pipeline")
            print("=" * 60)

        # Stage 1: Parse diff
        if not self.silent:
            print("\n[Stage 1] Parsing diff...")
        diff = parse_unified_diff(diff_text)

        if not self.silent:
            print(f"  Files: {len(diff.files)}")
            print(f"  Additions: +{diff.additions}")
            print(f"  Deletions: -{diff.deletions}")

        # Stage 2: Budget check
        if not self.silent:
            print("\n[Stage 2] Checking ATP budget...")

        required_atp = len(diff.files) * self.config.atp_per_file

        if len(diff.files) > self.config.max_files_without_approval and not force_large:
            if not self.silent:
                print(f"  BLOCKED: PR has {len(diff.files)} files (max: {self.config.max_files_without_approval})")
                print("  Use --force-large to override")
            self._blocked_count += 1
            return ReviewResult(
                approved=False,
                summary=f"PR too large: {len(diff.files)} files exceeds limit of {self.config.max_files_without_approval}",
                findings=[ReviewFinding(
                    file="<pr>",
                    severity=ReviewSeverity.HIGH,
                    category=ReviewCategory.QUALITY,
                    title="PR size limit exceeded",
                    description=f"This PR modifies {len(diff.files)} files. Consider splitting into smaller PRs.",
                )],
                security_score=0.0,
                quality_score=0.0,
                files_reviewed=0,
                lines_reviewed=0,
                review_time_ms=(time.time() - start_time) * 1000,
            )

        # Consume ATP
        if not self.budget.consume(required_atp, operation="code_review"):
            if not self.silent:
                print(f"  BLOCKED: Insufficient ATP ({self.budget.atp} < {required_atp})")
            self._blocked_count += 1
            return ReviewResult(
                approved=False,
                summary="Insufficient budget for review",
                findings=[],
                security_score=0.0,
                quality_score=0.0,
                files_reviewed=0,
                lines_reviewed=0,
                review_time_ms=(time.time() - start_time) * 1000,
            )

        if not self.silent:
            print(f"  Consumed: {required_atp} ATP (remaining: {self.budget.atp})")

        # Stage 3: Membrane pre-filter (critical threats only)
        if not self.silent:
            print("\n[Stage 3] Membrane pre-filter...")

        # Combine all added code for membrane check
        all_additions = "\n".join(
            line[1:] for hunk in diff.hunks
            for line in hunk.lines if line.startswith("+")
        )
        filter_result = self.membrane.filter(Signal(content=all_additions))

        if not filter_result.allowed:
            if not self.silent:
                print(f"  BLOCKED by membrane: {filter_result.threat_level.name}")
                for sig in filter_result.matched_signatures:
                    print(f"    - {sig.description}")

            self._blocked_count += 1
            return ReviewResult(
                approved=False,
                summary="Critical security threat detected by membrane filter",
                findings=[
                    ReviewFinding(
                        file="<multiple>",
                        severity=ReviewSeverity.CRITICAL,
                        category=ReviewCategory.SECURITY,
                        title=sig.description,
                        description=f"Detected pattern: {sig.pattern[:50]}",
                    )
                    for sig in filter_result.matched_signatures
                ],
                security_score=0.0,
                quality_score=0.0,
                files_reviewed=len(diff.files),
                lines_reviewed=diff.total_changes,
                review_time_ms=(time.time() - start_time) * 1000,
            )

        if not self.silent:
            print("  Passed membrane filter")

        # Stage 4: CFFL Gate (Security AND Quality)
        if not self.silent:
            print("\n[Stage 4] CFFL Gate (Security + Quality review)...")

        security_result = self.security_reviewer.review(diff)
        quality_result = self.quality_reviewer.review(diff)

        if not self.silent:
            print(f"  Security: {'APPROVED' if security_result.approved else 'BLOCKED'} (score: {security_result.score:.2f})")
            print(f"  Quality:  {'APPROVED' if quality_result.approved else 'BLOCKED'} (score: {quality_result.score:.2f})")

        # CFFL AND gate
        if self.config.require_both_approvals:
            approved = security_result.approved and quality_result.approved
        else:
            # OR gate (either can approve)
            approved = security_result.approved or quality_result.approved

        if not self.silent:
            gate_type = "AND" if self.config.require_both_approvals else "OR"
            print(f"  Gate ({gate_type}): {'APPROVED' if approved else 'BLOCKED'}")

        # Combine findings
        all_findings = security_result.findings + quality_result.findings

        # Stage 5: Chaperone validation
        if not self.silent:
            print("\n[Stage 5] Chaperone validation...")

        # Build the result
        result = ReviewResult(
            approved=approved,
            summary=self._generate_summary(approved, security_result, quality_result),
            findings=all_findings,
            security_score=security_result.score,
            quality_score=quality_result.score,
            files_reviewed=len(diff.files),
            lines_reviewed=diff.total_changes,
            review_time_ms=(time.time() - start_time) * 1000,
        )

        # Validate with Chaperone
        validation = self.chaperone.fold(result.model_dump_json(), ReviewResult)

        if not validation.valid:
            if not self.silent:
                print(f"  Chaperone validation failed: {validation.error_trace}")
            # This shouldn't happen if we built the result correctly
            result.summary = f"Internal error: {validation.error_trace}"
            result.approved = False
        else:
            if not self.silent:
                print("  Validation passed")

        # Update statistics
        if approved:
            self._approved_count += 1
        else:
            self._blocked_count += 1

        # Callback
        if self.config.on_review_complete:
            self.config.on_review_complete(result)

        return result

    def _generate_summary(
        self,
        approved: bool,
        security: ReviewerResult,
        quality: ReviewerResult,
    ) -> str:
        """Generate a human-readable summary."""
        if approved:
            return f"Review passed. Security: {security.score:.0%}, Quality: {quality.score:.0%}"

        issues = []
        if not security.approved:
            critical = sum(1 for f in security.findings if f.severity == ReviewSeverity.CRITICAL)
            high = sum(1 for f in security.findings if f.severity == ReviewSeverity.HIGH)
            issues.append(f"Security issues: {critical} critical, {high} high")

        if not quality.approved:
            issues.append(f"Quality score too low: {quality.score:.0%}")

        return "Review failed. " + "; ".join(issues)

    def format_github_comment(self, result: ReviewResult) -> str:
        """Format the review result as a GitHub PR comment."""
        lines = []

        # Header
        if result.approved:
            lines.append("## :white_check_mark: Code Review Passed")
        else:
            lines.append("## :x: Code Review Failed")

        lines.append("")
        lines.append(f"**Summary**: {result.summary}")
        lines.append("")

        # Scores
        lines.append("### Scores")
        lines.append(f"- Security: {result.security_score:.0%}")
        lines.append(f"- Quality: {result.quality_score:.0%}")
        lines.append(f"- Files reviewed: {result.files_reviewed}")
        lines.append(f"- Lines reviewed: {result.lines_reviewed}")
        lines.append("")

        # Findings
        if result.findings:
            lines.append("### Findings")
            lines.append("")

            # Group by severity
            for severity in [ReviewSeverity.CRITICAL, ReviewSeverity.HIGH,
                           ReviewSeverity.MEDIUM, ReviewSeverity.LOW, ReviewSeverity.INFO]:
                findings = [f for f in result.findings if f.severity == severity]
                if findings:
                    icon = {
                        ReviewSeverity.CRITICAL: ":rotating_light:",
                        ReviewSeverity.HIGH: ":warning:",
                        ReviewSeverity.MEDIUM: ":large_orange_diamond:",
                        ReviewSeverity.LOW: ":small_blue_diamond:",
                        ReviewSeverity.INFO: ":information_source:",
                    }.get(severity, "")

                    lines.append(f"#### {icon} {severity.value.upper()}")
                    for finding in findings:
                        loc = f"`{finding.file}"
                        if finding.line:
                            loc += f":{finding.line}"
                        loc += "`"
                        lines.append(f"- **{finding.title}** ({loc})")
                        lines.append(f"  {finding.description}")
                        if finding.suggestion:
                            lines.append(f"  > Suggestion: {finding.suggestion}")
                    lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Review completed in {result.review_time_ms:.0f}ms by Operon Code Review Pipeline*")

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_reviews": self._reviews_count,
            "approved": self._approved_count,
            "blocked": self._blocked_count,
            "approval_rate": (
                self._approved_count / self._reviews_count
                if self._reviews_count > 0 else 0.0
            ),
            "budget_remaining": self.budget.atp,
            "budget_state": self.budget.get_state().value,
        }


# =============================================================================
# Demo and Tests
# =============================================================================

MOCK_DIFF_CLEAN = """
diff --git a/src/utils.py b/src/utils.py
new file mode 100644
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,15 @@
+\"\"\"Utility functions.\"\"\"
+
+def add_numbers(a: int, b: int) -> int:
+    \"\"\"Add two numbers safely.\"\"\"
+    return a + b
+
+def format_name(first: str, last: str) -> str:
+    \"\"\"Format a full name.\"\"\"
+    return f"{first} {last}"
+
+def validate_email(email: str) -> bool:
+    \"\"\"Validate email format.\"\"\"
+    import re
+    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
+    return bool(re.match(pattern, email))
"""

MOCK_DIFF_SECURITY_ISSUES = """
diff --git a/src/dangerous.py b/src/dangerous.py
new file mode 100644
--- /dev/null
+++ b/src/dangerous.py
@@ -0,0 +1,20 @@
+\"\"\"This code has security issues.\"\"\"
+import os
+import pickle
+
+API_KEY = "sk-1234567890abcdef"  # Hardcoded secret!
+
+def run_command(cmd: str) -> str:
+    \"\"\"Run a shell command.\"\"\"
+    return os.system(cmd)  # Shell injection risk
+
+def process_data(data: bytes) -> dict:
+    \"\"\"Load pickled data.\"\"\"
+    return pickle.loads(data)  # Unsafe deserialization
+
+def execute_code(code: str) -> None:
+    \"\"\"Execute arbitrary code.\"\"\"
+    eval(code)  # Code injection!
+
+def get_file(path: str) -> str:
+    return open("../../../etc/passwd").read()  # Path traversal
"""

MOCK_DIFF_QUALITY_ISSUES = """
diff --git a/src/messy.py b/src/messy.py
new file mode 100644
--- /dev/null
+++ b/src/messy.py
@@ -0,0 +1,15 @@
+\"\"\"Code with quality issues.\"\"\"
+
+def process():
+    try:
+        do_something()
+    except:
+        pass  # Bare except!
+
+def magic():
+    x = 12345678  # Magic number
+    return x * 9876543
+
+def debug():
+    print("This should use logging")  # TODO: fix this
+    return True
"""


def run_demo():
    """Run demonstration with mock diffs."""
    print("=" * 70)
    print("Example 43: Code Review Pipeline - Demo")
    print("=" * 70)

    pipeline = CodeReviewPipeline(
        config=PipelineConfig(
            total_budget=2000,
            atp_per_file=100,
        ),
        silent=False,
    )

    # Test 1: Clean code
    print("\n" + "=" * 70)
    print("TEST 1: Clean Code (should PASS)")
    print("=" * 70)
    result1 = pipeline.review(MOCK_DIFF_CLEAN)
    print(f"\nResult: {'APPROVED' if result1.approved else 'BLOCKED'}")
    print(f"Summary: {result1.summary}")

    # Test 2: Security issues
    print("\n" + "=" * 70)
    print("TEST 2: Security Issues (should FAIL)")
    print("=" * 70)
    result2 = pipeline.review(MOCK_DIFF_SECURITY_ISSUES)
    print(f"\nResult: {'APPROVED' if result2.approved else 'BLOCKED'}")
    print(f"Summary: {result2.summary}")
    print(f"Findings: {len(result2.findings)}")
    for f in result2.findings[:3]:
        print(f"  - [{f.severity.value}] {f.title}")

    # Test 3: Quality issues
    print("\n" + "=" * 70)
    print("TEST 3: Quality Issues (may PASS with warnings)")
    print("=" * 70)
    result3 = pipeline.review(MOCK_DIFF_QUALITY_ISSUES)
    print(f"\nResult: {'APPROVED' if result3.approved else 'BLOCKED'}")
    print(f"Summary: {result3.summary}")

    # Show GitHub comment format
    print("\n" + "=" * 70)
    print("GitHub Comment Format (Test 2):")
    print("=" * 70)
    print(pipeline.format_github_comment(result2))

    # Statistics
    print("\n" + "=" * 70)
    print("Pipeline Statistics")
    print("=" * 70)
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def run_smoke_test():
    """Automated smoke test for CI."""
    print("Running smoke tests...\n")

    pipeline = CodeReviewPipeline(silent=True)

    # Test 1: Clean code should pass
    result = pipeline.review(MOCK_DIFF_CLEAN)
    assert result.approved, f"Clean code should pass, got: {result.summary}"
    assert result.security_score >= 0.9, f"Security score should be high: {result.security_score}"
    print("  Test 1: Clean code review - PASSED")

    # Test 2: Security issues should fail
    result = pipeline.review(MOCK_DIFF_SECURITY_ISSUES)
    assert not result.approved, "Security issues should fail"
    assert result.security_score < 0.5, f"Security score should be low: {result.security_score}"
    assert len(result.findings) > 0, "Should have findings"
    print("  Test 2: Security issue detection - PASSED")

    # Test 3: Diff parsing
    diff = parse_unified_diff(MOCK_DIFF_CLEAN)
    assert len(diff.files) == 1, f"Should have 1 file, got {len(diff.files)}"
    assert diff.additions > 0, "Should have additions"
    print("  Test 3: Diff parsing - PASSED")

    # Test 4: Schema validation
    result = ReviewResult(
        approved=True,
        summary="Test",
        findings=[],
        security_score=1.0,
        quality_score=1.0,
        files_reviewed=1,
        lines_reviewed=10,
    )
    assert result.approved
    print("  Test 4: Schema validation - PASSED")

    # Test 5: Budget enforcement
    pipeline2 = CodeReviewPipeline(
        config=PipelineConfig(total_budget=50, atp_per_file=100),
        silent=True,
    )
    result = pipeline2.review(MOCK_DIFF_CLEAN)
    assert not result.approved, "Should fail due to insufficient budget"
    print("  Test 5: Budget enforcement - PASSED")

    print("\nSmoke tests passed!")


def main():
    """Main entry point."""
    if "--test" in sys.argv:
        run_smoke_test()
    elif "--pr" in sys.argv:
        # Read diff from file
        idx = sys.argv.index("--pr")
        if idx + 1 < len(sys.argv):
            with open(sys.argv[idx + 1]) as f:
                diff_text = f.read()
            pipeline = CodeReviewPipeline(silent=False)
            result = pipeline.review(diff_text, force_large="--force-large" in sys.argv)
            print("\n" + pipeline.format_github_comment(result))
        else:
            print("Error: --pr requires a diff file path")
            sys.exit(1)
    else:
        run_demo()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        raise
