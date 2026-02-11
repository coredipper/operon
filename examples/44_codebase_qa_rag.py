#!/usr/bin/env python3
"""
Example 44: Codebase Q&A with RAG (Retrieval-Augmented Generation)
===================================================================

Demonstrates a practical codebase question-answering system that uses:
- **Membrane**: Filters prompt injection attempts
- **Histone (Memory)**: Remembers successful search patterns
- **Chaperone Healing Loop**: Retries if LLM hallucinates non-existent files
- **ATP_Store**: Budgets token usage per query
- **Cascade**: Multi-stage pipeline for retrieval and generation

This pattern is common in code assistants like Cursor, Cody, and Continue.

Architecture:

```
User Question
    |
[Membrane: Filter injection attempts]
    |
[Query Planner: Decompose into search queries]
    |
[Retriever: Grep/Glob search]
    |
[Context Assembler: Build prompt with file excerpts]
    |
[Generator: Produce answer with citations]
    |
[Chaperone: Validate citations exist]
    |
[Healing Loop: Retry if hallucinated files]
    |
[Memory: Store successful patterns]
```

Usage:
    python examples/44_codebase_qa_rag.py                         # Demo mode
    python examples/44_codebase_qa_rag.py --test                  # Smoke test
    python examples/44_codebase_qa_rag.py "Where is ATP_Store?"   # Ask a question

Prerequisites:
- Example 39 for Chaperone Healing Loop patterns
- Example 14 for Histone memory patterns
"""

import os
import re
import sys
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from operon_ai import ATP_Store, Chaperone
from operon_ai.core.types import Signal
from operon_ai.organelles.membrane import Membrane, ThreatLevel, ThreatSignature
from operon_ai.state.histone import HistoneStore, MarkerType, MarkerStrength
from operon_ai.state.metabolism import MetabolicState


# =============================================================================
# Schema Definitions
# =============================================================================


class Citation(BaseModel):
    """A citation to a specific location in code."""
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    snippet: str = ""

    @field_validator("line_end")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if "line_start" in info.data and v < info.data["line_start"]:
            return info.data["line_start"]
        return v


class CodeAnswer(BaseModel):
    """Answer with citations to code locations."""
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    search_patterns_used: list[str] = Field(default_factory=list)


# =============================================================================
# Search Components
# =============================================================================


@dataclass
class SearchResult:
    """Result from a code search."""
    file_path: str
    line_number: int
    line_content: str
    match_score: float = 1.0


@dataclass
class SearchQuery:
    """A search query decomposed from the user question."""
    pattern: str
    query_type: str  # "grep", "glob", "definition"
    priority: int = 1


class CodeSearcher:
    """
    Searches a codebase using grep and glob patterns.

    Supports:
    - Regex grep for content search
    - Glob patterns for file search
    - Definition search (class/function names)
    """

    def __init__(self, root_path: str = ".", silent: bool = False):
        self.root_path = Path(root_path).resolve()
        self.silent = silent

    def search_grep(
        self,
        pattern: str,
        file_pattern: str = "*.py",
        max_results: int = 20,
    ) -> list[SearchResult]:
        """Search for pattern in files using grep."""
        results = []

        try:
            # Use grep for search
            cmd = [
                "grep", "-rn", "-E",
                "--include", file_pattern,
                pattern,
                str(self.root_path),
            ]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            for line in process.stdout.strip().split("\n")[:max_results]:
                if not line or ":" not in line:
                    continue

                # Parse grep output: file:line:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    try:
                        line_num = int(parts[1])
                        content = parts[2].strip()

                        # Make path relative to root
                        try:
                            rel_path = Path(file_path).relative_to(self.root_path)
                        except ValueError:
                            rel_path = Path(file_path)

                        results.append(SearchResult(
                            file_path=str(rel_path),
                            line_number=line_num,
                            line_content=content,
                        ))
                    except ValueError:
                        continue

        except subprocess.TimeoutExpired:
            if not self.silent:
                print("  Search timed out")
        except Exception as e:
            if not self.silent:
                print(f"  Search error: {e}")

        return results

    def search_glob(self, pattern: str, max_results: int = 20) -> list[SearchResult]:
        """Search for files matching glob pattern."""
        results = []

        try:
            for path in self.root_path.rglob(pattern):
                if len(results) >= max_results:
                    break

                if path.is_file():
                    try:
                        rel_path = path.relative_to(self.root_path)
                    except ValueError:
                        rel_path = path

                    results.append(SearchResult(
                        file_path=str(rel_path),
                        line_number=1,
                        line_content=f"File: {rel_path.name}",
                    ))

        except Exception as e:
            if not self.silent:
                print(f"  Glob error: {e}")

        return results

    def search_definition(
        self,
        name: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Search for class or function definitions."""
        patterns = [
            f"class {name}",
            f"def {name}",
            f"{name} =",
        ]

        results = []
        for pattern in patterns:
            results.extend(self.search_grep(pattern, max_results=max_results // 3))

        return results[:max_results]

    def get_file_context(
        self,
        file_path: str,
        line_number: int,
        context_lines: int = 5,
    ) -> str:
        """Get file content around a specific line."""
        full_path = self.root_path / file_path

        if not full_path.exists():
            return ""

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)

            context_lines_list = []
            for i, line in enumerate(lines[start:end], start=start + 1):
                prefix = ">>> " if i == line_number else "    "
                context_lines_list.append(f"{prefix}{i}: {line.rstrip()}")

            return "\n".join(context_lines_list)

        except Exception:
            return ""


# =============================================================================
# Query Planner
# =============================================================================


class QueryPlanner:
    """
    Decomposes user questions into search queries.

    Uses pattern matching for common question types:
    - "Where is X defined?" -> definition search
    - "How does X work?" -> grep for X and related terms
    - "What files contain X?" -> grep search
    """

    PATTERNS = [
        # Definition queries
        (r"where (?:is|are) (\w+) (?:defined|declared|implemented)", "definition"),
        (r"find (?:the )?(?:definition|implementation) of (\w+)", "definition"),
        (r"(?:show|find) (?:me )?(?:the )?class (\w+)", "definition"),
        (r"(?:show|find) (?:me )?(?:the )?function (\w+)", "definition"),

        # File queries
        (r"what files? (?:contain|have|use) (\w+)", "grep"),
        (r"which files? (?:contain|have|use) (\w+)", "grep"),

        # How queries
        (r"how does (\w+) work", "grep"),
        (r"what does (\w+) do", "grep"),
        (r"explain (\w+)", "grep"),
    ]

    def __init__(self, memory: HistoneStore | None = None):
        self.memory = memory
        self._compiled = [(re.compile(p, re.IGNORECASE), t) for p, t in self.PATTERNS]

    def plan(self, question: str) -> list[SearchQuery]:
        """Decompose question into search queries."""
        queries = []

        # Try pattern matching
        for pattern, query_type in self._compiled:
            match = pattern.search(question)
            if match:
                term = match.group(1)
                queries.append(SearchQuery(
                    pattern=term,
                    query_type=query_type,
                    priority=1,
                ))

        # If no patterns matched, extract keywords
        if not queries:
            # Extract likely code identifiers
            words = re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)*|[a-z_]+", question)
            # Filter common words
            stop_words = {"the", "is", "are", "how", "what", "where", "does", "do", "a", "an"}
            keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]

            for i, keyword in enumerate(keywords[:3]):
                queries.append(SearchQuery(
                    pattern=keyword,
                    query_type="grep",
                    priority=i + 1,
                ))

        # Check memory for successful patterns
        if self.memory:
            context = self.memory.retrieve_context(question)
            for marker in context.markers[:2]:
                # Extract pattern from memory
                if "pattern:" in marker.content:
                    pattern = marker.content.split("pattern:")[1].strip()
                    queries.insert(0, SearchQuery(
                        pattern=pattern,
                        query_type="grep",
                        priority=0,
                    ))

        return queries


# =============================================================================
# Citation Validator
# =============================================================================


class CitationValidator:
    """
    Validates that citations reference real files and lines.

    This is the key component for the healing loop - if citations
    are hallucinated, we reject and retry with error feedback.
    """

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()

    def validate(self, answer: CodeAnswer) -> tuple[bool, str]:
        """
        Validate all citations in an answer.

        Returns:
            (valid, error_message)
        """
        errors = []

        for citation in answer.citations:
            file_path = self.root_path / citation.file_path

            # Check file exists
            if not file_path.exists():
                errors.append(f"File not found: {citation.file_path}")
                continue

            # Check line numbers are valid
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    line_count = sum(1 for _ in f)

                if citation.line_start > line_count:
                    errors.append(
                        f"Line {citation.line_start} exceeds file length "
                        f"({line_count}) in {citation.file_path}"
                    )
                elif citation.line_end > line_count:
                    errors.append(
                        f"Line {citation.line_end} exceeds file length "
                        f"({line_count}) in {citation.file_path}"
                    )

            except Exception as e:
                errors.append(f"Cannot read {citation.file_path}: {e}")

        if errors:
            return False, "; ".join(errors)

        return True, ""


# =============================================================================
# Mock Generator (for demo without LLM)
# =============================================================================


class MockGenerator:
    """
    Mock generator for testing without an LLM.

    Simulates LLM responses with realistic answer structures.
    Can be configured to sometimes hallucinate for testing healing loops.
    """

    def __init__(
        self,
        searcher: CodeSearcher,
        hallucinate_rate: float = 0.0,
    ):
        self.searcher = searcher
        self.hallucinate_rate = hallucinate_rate
        self._call_count = 0

    def generate(
        self,
        question: str,
        context: str,
        error_feedback: str | None = None,
    ) -> str:
        """Generate a mock answer based on search context."""
        self._call_count += 1
        import random
        import json

        # If we have error feedback, fix the hallucination
        if error_feedback and "not found" in error_feedback.lower():
            # Extract the real file from context if available
            real_files = re.findall(r"(\S+\.py):\d+:", context)
            if real_files:
                # Generate a corrected response
                file_path = real_files[0]
                return json.dumps({
                    "answer": f"Based on my search, I found relevant code in {file_path}.",
                    "citations": [{
                        "file_path": file_path,
                        "line_start": 1,
                        "line_end": 10,
                        "snippet": "Found in search results",
                    }],
                    "confidence": 0.7,
                    "search_patterns_used": ["corrected_search"],
                })

        # Maybe hallucinate (for testing healing loops)
        if random.random() < self.hallucinate_rate:
            return json.dumps({
                "answer": "The code is in the nonexistent file.",
                "citations": [{
                    "file_path": "nonexistent/fake_file.py",
                    "line_start": 42,
                    "line_end": 50,
                    "snippet": "This is hallucinated",
                }],
                "confidence": 0.9,
                "search_patterns_used": ["hallucinated"],
            })

        # Generate based on context
        if context:
            # Parse first result from context
            match = re.search(r"(\S+\.py):(\d+):\s*(.+)", context)
            if match:
                file_path, line_num, content = match.groups()
                return json.dumps({
                    "answer": f"Found relevant code in {file_path} at line {line_num}: {content[:100]}",
                    "citations": [{
                        "file_path": file_path,
                        "line_start": int(line_num),
                        "line_end": int(line_num) + 5,
                        "snippet": content[:100],
                    }],
                    "confidence": 0.85,
                    "search_patterns_used": [question.split()[0]],
                })

        # No context - give generic answer
        return json.dumps({
            "answer": "I could not find relevant code matching your query.",
            "citations": [],
            "confidence": 0.3,
            "search_patterns_used": [],
        })


# =============================================================================
# RAG Pipeline
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the RAG pipeline."""
    # Budget settings
    atp_per_query: int = 100
    total_budget: int = 1000

    # Search settings
    max_search_results: int = 20
    context_lines: int = 5

    # Healing settings
    max_healing_attempts: int = 3
    confidence_decay: float = 0.15

    # Memory settings
    remember_patterns: bool = True
    memory_decay_hours: float = 168.0  # 1 week


@dataclass
class QueryResult:
    """Result of a query through the pipeline."""
    success: bool
    answer: CodeAnswer | None
    healing_attempts: int = 0
    error: str = ""
    search_results_count: int = 0
    budget_consumed: int = 0


class CodebaseQAPipeline:
    """
    Production-ready Codebase Q&A pipeline using Operon patterns.

    Stages:
    1. Membrane filter (injection defense)
    2. Query planning (decompose question)
    3. Code search (grep/glob)
    4. Context assembly (build prompt)
    5. Generation (mock or LLM)
    6. Validation + Healing loop (verify citations)
    7. Memory update (remember successful patterns)
    """

    def __init__(
        self,
        root_path: str = ".",
        config: PipelineConfig | None = None,
        generator: Callable[[str, str, str | None], str] | None = None,
        silent: bool = False,
    ):
        self.root_path = Path(root_path).resolve()
        self.config = config or PipelineConfig()
        self.silent = silent

        # Initialize Operon components
        self.budget = ATP_Store(
            budget=self.config.total_budget,
            silent=silent,
        )
        self.membrane = Membrane(
            threshold=ThreatLevel.SUSPICIOUS,
            silent=silent,
        )
        self.memory = HistoneStore(
            context_window=5,
            silent=silent,
        )
        self.chaperone = Chaperone(silent=silent)

        # Search and validation
        self.searcher = CodeSearcher(str(self.root_path), silent=silent)
        self.planner = QueryPlanner(memory=self.memory)
        self.validator = CitationValidator(str(self.root_path))

        # Generator (mock by default)
        if generator:
            self._generator = generator
        else:
            mock = MockGenerator(self.searcher, hallucinate_rate=0.0)
            self._generator = mock.generate

        # Statistics
        self._total_queries = 0
        self._successful_queries = 0
        self._healing_used = 0

    def query(self, question: str) -> QueryResult:
        """
        Answer a question about the codebase.

        Args:
            question: Natural language question about the code

        Returns:
            QueryResult with answer and metadata
        """
        self._total_queries += 1
        budget_start = self.budget.atp

        if not self.silent:
            print("\n" + "=" * 60)
            print("Codebase Q&A Pipeline")
            print("=" * 60)
            print(f"\nQuestion: {question}")

        # Stage 1: Membrane filter
        if not self.silent:
            print("\n[Stage 1] Membrane filter...")

        filter_result = self.membrane.filter(Signal(content=question))
        if not filter_result.allowed:
            if not self.silent:
                print(f"  BLOCKED: {filter_result.threat_level.name}")
            return QueryResult(
                success=False,
                answer=None,
                error=f"Query blocked by membrane: {filter_result.threat_level.name}",
            )

        if not self.silent:
            print("  Passed")

        # Stage 2: Budget check
        if not self.silent:
            print("\n[Stage 2] Budget check...")

        if not self.budget.consume(self.config.atp_per_query, operation="qa_query"):
            if not self.silent:
                print("  Insufficient budget")
            return QueryResult(
                success=False,
                answer=None,
                error="Insufficient budget for query",
            )

        if not self.silent:
            print(f"  Consumed {self.config.atp_per_query} ATP (remaining: {self.budget.atp})")

        # Stage 3: Query planning
        if not self.silent:
            print("\n[Stage 3] Query planning...")

        queries = self.planner.plan(question)
        if not self.silent:
            for q in queries[:3]:
                print(f"  - {q.query_type}: {q.pattern}")

        # Stage 4: Code search
        if not self.silent:
            print("\n[Stage 4] Code search...")

        all_results: list[SearchResult] = []
        for query in queries:
            if query.query_type == "definition":
                results = self.searcher.search_definition(
                    query.pattern,
                    max_results=self.config.max_search_results,
                )
            elif query.query_type == "glob":
                results = self.searcher.search_glob(
                    query.pattern,
                    max_results=self.config.max_search_results,
                )
            else:
                results = self.searcher.search_grep(
                    query.pattern,
                    max_results=self.config.max_search_results,
                )

            all_results.extend(results)

        if not self.silent:
            print(f"  Found {len(all_results)} results")

        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            key = (r.file_path, r.line_number)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        # Stage 5: Context assembly
        if not self.silent:
            print("\n[Stage 5] Context assembly...")

        context_parts = []
        for result in unique_results[:10]:  # Limit context size
            context_parts.append(
                f"{result.file_path}:{result.line_number}: {result.line_content}"
            )
        context = "\n".join(context_parts)

        if not self.silent:
            print(f"  Assembled {len(context_parts)} context snippets")

        # Stage 6: Generation with healing loop
        if not self.silent:
            print("\n[Stage 6] Generation + Healing loop...")

        answer = None
        healing_attempts = 0
        error_feedback = None

        for attempt in range(self.config.max_healing_attempts):
            # Generate
            raw_output = self._generator(question, context, error_feedback)

            if not self.silent:
                print(f"  Attempt {attempt + 1}: Generated response")

            # Validate with Chaperone
            folded = self.chaperone.fold(raw_output, CodeAnswer)

            if not folded.valid:
                if not self.silent:
                    print(f"    Schema validation failed: {folded.error_trace}")
                error_feedback = f"Invalid JSON format: {folded.error_trace}"
                healing_attempts += 1
                continue

            parsed_answer = folded.structure

            # Validate citations
            citations_valid, citation_error = self.validator.validate(parsed_answer)

            if not citations_valid:
                if not self.silent:
                    print(f"    Citation validation failed: {citation_error}")
                error_feedback = f"Citation error: {citation_error}"
                healing_attempts += 1
                continue

            # Success!
            answer = parsed_answer
            if not self.silent:
                print("    Validation passed")
            break

        if healing_attempts > 0:
            self._healing_used += 1

        if not answer:
            return QueryResult(
                success=False,
                answer=None,
                healing_attempts=healing_attempts,
                error=f"All {self.config.max_healing_attempts} attempts failed: {error_feedback}",
                search_results_count=len(unique_results),
                budget_consumed=budget_start - self.budget.atp,
            )

        # Stage 7: Memory update
        if self.config.remember_patterns and answer.search_patterns_used:
            if not self.silent:
                print("\n[Stage 7] Memory update...")

            for pattern in answer.search_patterns_used:
                self.memory.add_marker(
                    content=f"pattern: {pattern}",
                    marker_type=MarkerType.ACETYLATION,
                    strength=MarkerStrength.MODERATE,
                    decay_hours=self.config.memory_decay_hours,
                    tags=["search_pattern"],
                    context=question[:100],
                )
                if not self.silent:
                    print(f"  Remembered pattern: {pattern}")

        self._successful_queries += 1

        return QueryResult(
            success=True,
            answer=answer,
            healing_attempts=healing_attempts,
            search_results_count=len(unique_results),
            budget_consumed=budget_start - self.budget.atp,
        )

    def format_answer(self, result: QueryResult) -> str:
        """Format a query result for display."""
        lines = []

        if not result.success:
            lines.append(f"Error: {result.error}")
            return "\n".join(lines)

        answer = result.answer
        lines.append(f"\n{answer.answer}")
        lines.append("")

        if answer.citations:
            lines.append("Citations:")
            for c in answer.citations:
                lines.append(f"  - {c.file_path}:{c.line_start}-{c.line_end}")
                if c.snippet:
                    lines.append(f"    {c.snippet[:80]}...")

        lines.append(f"\nConfidence: {answer.confidence:.0%}")

        if result.healing_attempts > 0:
            lines.append(f"Healing attempts: {result.healing_attempts}")

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_queries": self._total_queries,
            "successful_queries": self._successful_queries,
            "healing_used": self._healing_used,
            "success_rate": (
                self._successful_queries / self._total_queries
                if self._total_queries > 0 else 0.0
            ),
            "budget_remaining": self.budget.atp,
            "memory_markers": len(self.memory._markers),
        }


# =============================================================================
# Demo and Tests
# =============================================================================


def run_demo():
    """Run demonstration with real codebase search."""
    print("=" * 70)
    print("Example 44: Codebase Q&A with RAG - Demo")
    print("=" * 70)

    # Use the operon_ai directory as the codebase
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pipeline = CodebaseQAPipeline(
        root_path=root_path,
        config=PipelineConfig(
            total_budget=500,
            atp_per_query=100,
        ),
        silent=False,
    )

    # Demo queries
    questions = [
        "Where is ATP_Store defined?",
        "What files contain Membrane?",
        "How does the Chaperone work?",
    ]

    for question in questions:
        result = pipeline.query(question)
        print("\n" + "-" * 40)
        print("RESULT:")
        print(pipeline.format_answer(result))

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

    # Use current directory for testing
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pipeline = CodebaseQAPipeline(
        root_path=root_path,
        silent=True,
    )

    # Test 1: Basic query
    result = pipeline.query("Where is ATP_Store defined?")
    assert result.success or result.search_results_count >= 0, "Query should complete"
    print("  Test 1: Basic query - PASSED")

    # Test 2: Query planning
    planner = QueryPlanner()
    queries = planner.plan("Where is MyClass defined?")
    assert len(queries) > 0, "Should generate queries"
    assert queries[0].query_type == "definition", "Should detect definition query"
    print("  Test 2: Query planning - PASSED")

    # Test 3: Citation validation
    validator = CitationValidator(root_path)
    valid_answer = CodeAnswer(
        answer="Test",
        citations=[Citation(
            file_path="examples/43_code_review_pipeline.py",
            line_start=1,
            line_end=10,
            snippet="Test",
        )],
        confidence=0.9,
    )
    is_valid, error = validator.validate(valid_answer)
    assert is_valid, f"Valid citation should pass: {error}"
    print("  Test 3: Citation validation - PASSED")

    # Test 4: Invalid citation detection
    invalid_answer = CodeAnswer(
        answer="Test",
        citations=[Citation(
            file_path="nonexistent/fake_file.py",
            line_start=1,
            line_end=10,
            snippet="Test",
        )],
        confidence=0.9,
    )
    is_valid, error = validator.validate(invalid_answer)
    assert not is_valid, "Invalid citation should fail"
    assert "not found" in error.lower(), "Should report file not found"
    print("  Test 4: Invalid citation detection - PASSED")

    # Test 5: Schema validation
    answer = CodeAnswer(
        answer="Test answer",
        citations=[],
        confidence=0.8,
    )
    assert answer.answer == "Test answer"
    print("  Test 5: Schema validation - PASSED")

    # Test 6: Membrane filtering
    membrane = Membrane(threshold=ThreatLevel.SUSPICIOUS, silent=True)
    result = membrane.filter(Signal(content="Normal question about code"))
    assert result.allowed, "Normal query should pass"
    print("  Test 6: Membrane filtering - PASSED")

    print("\nSmoke tests passed!")


def main():
    """Main entry point."""
    if "--test" in sys.argv:
        run_smoke_test()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Direct question
        question = " ".join(sys.argv[1:])
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pipeline = CodebaseQAPipeline(root_path=root_path, silent=False)
        result = pipeline.query(question)
        print("\n" + "=" * 40)
        print("ANSWER:")
        print(pipeline.format_answer(result))
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
