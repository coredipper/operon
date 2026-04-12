"""Integration tests for DeerFlow end-to-end execution.

Requires:
- DeerFlow harness installed (Python >=3.12)
- Ollama running with at least one model

Tests are skipped cleanly when either dependency is unavailable.
"""

from __future__ import annotations

import pytest

from operon_ai import ATP_Store, MockProvider, Nucleus, SkillStage, skill_organism
from operon_ai.convergence.deerflow_compiler import organism_to_deerflow
from operon_ai.convergence.deerflow_executor import (
    DeerFlowResult,
    HAS_DEERFLOW,
    _compiled_to_agent_kwargs,
    execute_deerflow,
)

# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

OLLAMA_AVAILABLE = False
try:
    import urllib.request

    req = urllib.request.Request("http://localhost:11434/", method="HEAD")
    urllib.request.urlopen(req, timeout=2)
    OLLAMA_AVAILABLE = True
except Exception:
    pass

needs_deerflow = pytest.mark.skipif(
    not HAS_DEERFLOW, reason="DeerFlow not installed"
)
needs_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE, reason="Ollama not running"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_organism():
    """Build a simple 2-stage organism for testing."""
    provider = MockProvider(responses={})
    nucleus = Nucleus(provider=provider)
    return skill_organism(
        stages=[
            SkillStage(
                name="researcher",
                role="Researcher",
                instructions="Search for relevant information. Summarize findings.",
                mode="fixed",
            ),
            SkillStage(
                name="writer",
                role="Writer",
                instructions="Write a clear response based on the research.",
                mode="fuzzy",
            ),
        ],
        fast_nucleus=nucleus,
        deep_nucleus=nucleus,
        budget=ATP_Store(budget=1000),
    )


# ---------------------------------------------------------------------------
# Unit-level tests (no DeerFlow needed)
# ---------------------------------------------------------------------------


class TestSchemaTransformation:
    """Verify _compiled_to_agent_kwargs produces valid kwargs."""

    def test_basic_transformation(self):
        org = _make_organism()
        compiled = organism_to_deerflow(org)
        kwargs = _compiled_to_agent_kwargs(compiled)

        assert kwargs["name"] == "researcher"
        assert "system_prompt" in kwargs
        assert isinstance(kwargs["sandbox"], bool)
        assert kwargs["subagent"] is False

    def test_system_prompt_includes_skills(self):
        org = _make_organism()
        compiled = organism_to_deerflow(org)
        kwargs = _compiled_to_agent_kwargs(compiled)

        assert "Search for relevant information" in kwargs["system_prompt"]

    def test_system_prompt_includes_sub_agents(self):
        org = _make_organism()
        compiled = organism_to_deerflow(org)
        kwargs = _compiled_to_agent_kwargs(compiled)

        # writer is a sub-agent in DeerFlow compilation
        assert "writer" in kwargs["system_prompt"].lower()

    def test_sandbox_mapping(self):
        org = _make_organism()
        compiled = organism_to_deerflow(org)

        # Default sandbox is "none"
        kwargs = _compiled_to_agent_kwargs(compiled)
        assert kwargs["sandbox"] is False

        # Docker sandbox
        compiled_docker = dict(compiled, sandbox="docker")
        kwargs_docker = _compiled_to_agent_kwargs(compiled_docker)
        assert kwargs_docker["sandbox"] is True

    def test_certificates_not_in_kwargs(self):
        """Certificates are verified post-execution, not passed to DeerFlow."""
        org = _make_organism()
        compiled = organism_to_deerflow(org)
        kwargs = _compiled_to_agent_kwargs(compiled)

        assert "certificates" not in kwargs


# ---------------------------------------------------------------------------
# Integration tests (need DeerFlow + Ollama)
# ---------------------------------------------------------------------------


@needs_deerflow
@needs_ollama
class TestDeerflowExecution:
    """End-to-end execution against real DeerFlow + Ollama."""

    def test_single_task_produces_output(self):
        """Compile organism, execute in DeerFlow, get non-empty output."""
        org = _make_organism()
        compiled = organism_to_deerflow(org)

        result = execute_deerflow(
            compiled,
            task="What is 2 + 2? Answer in one word.",
            model_name="gemma4:latest",
            timeout_seconds=60.0,
        )

        assert isinstance(result, DeerFlowResult)
        assert len(result.output) > 0
        assert result.timing_ms > 0

    def test_watcher_summary_populated(self):
        """Watcher observes execution and produces a summary."""
        org = _make_organism()
        compiled = organism_to_deerflow(org)

        result = execute_deerflow(
            compiled,
            task="Say hello.",
            model_name="gemma4:latest",
            enable_watcher=True,
            timeout_seconds=60.0,
        )

        assert result.watcher_summary is not None
        assert "total_stages" in result.watcher_summary

    def test_certificates_verified_after_execution(self):
        """Certificates from the compiled dict still verify after DeerFlow run."""
        org = _make_organism()
        compiled = organism_to_deerflow(org)

        result = execute_deerflow(
            compiled,
            task="Say hello.",
            model_name="gemma4:latest",
            verify_certificates=True,
            timeout_seconds=60.0,
        )

        # ATP budget certificate should hold (budget > 0)
        assert len(result.certificates_verified) >= 1
        for cert in result.certificates_verified:
            assert cert["holds"] is True, (
                f"Certificate {cert['theorem']} failed after DeerFlow execution"
            )

    def test_watcher_disabled(self):
        """Execution works without the watcher."""
        org = _make_organism()
        compiled = organism_to_deerflow(org)

        result = execute_deerflow(
            compiled,
            task="Say hello.",
            model_name="gemma4:latest",
            enable_watcher=False,
            timeout_seconds=60.0,
        )

        assert len(result.output) > 0
        assert result.watcher_summary == {}
