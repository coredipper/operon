"""Tests for CLI stage handler."""

import sys

from operon_ai import CLIResult, cli_handler, SkillStage, skill_organism
from operon_ai.organelles.nucleus import Nucleus
from operon_ai.providers import MockProvider


# ---------------------------------------------------------------------------
# cli_handler basics
# ---------------------------------------------------------------------------


def test_cli_handler_arg_mode():
    h = cli_handler("echo", input_mode="arg")
    result = h("hello world")
    assert result["output"] == "hello world"
    assert result["_action_type"] == "EXECUTE"
    assert result["cli_result"].returncode == 0


def test_cli_handler_stdin_mode():
    h = cli_handler("cat", input_mode="stdin")
    result = h("piped input")
    assert result["output"] == "piped input"


def test_cli_handler_none_mode():
    h = cli_handler("echo fixed output", input_mode="none")
    result = h("ignored")
    assert "fixed output" in result["output"]


def test_cli_handler_failure_action_type():
    h = cli_handler("false")
    result = h("")
    assert result["_action_type"] == "FAILURE"
    assert result["cli_result"].returncode != 0


def test_cli_handler_custom_success_codes():
    h = cli_handler("false", success_codes=(0, 1))
    result = h("")
    assert result["_action_type"] == "EXECUTE"  # false returns 1, but 1 is success


def test_cli_handler_timeout():
    h = cli_handler("sleep 10", input_mode="none", timeout=0.1)
    result = h("")
    assert result["cli_result"].timed_out is True
    assert result["_action_type"] == "FAILURE"


def test_cli_handler_parse_output():
    h = cli_handler("echo", parse_output=lambda s: s.strip().upper())
    result = h("hello")
    assert result["output"] == "HELLO"


def test_cli_result_is_frozen():
    h = cli_handler("echo")
    result = h("test")
    cli_result = result["cli_result"]
    assert isinstance(cli_result, CLIResult)
    assert cli_result.command  # non-empty


# ---------------------------------------------------------------------------
# _action_type in organism
# ---------------------------------------------------------------------------


def test_action_type_override_in_organism():
    """Handler returning _action_type=FAILURE triggers halt_on_block."""
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))

    organism = skill_organism(
        stages=[
            SkillStage(name="fail", role="Failer",
                       handler=cli_handler("false")),
            SkillStage(name="never", role="Unreachable",
                       handler=lambda t: "should not run"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
        halt_on_block=True,
    )
    result = organism.run("test")
    # Should halt after "fail" stage — "never" stage should not run
    assert len(result.stage_results) == 1
    assert result.stage_results[0].action_type == "FAILURE"


def test_action_type_execute_passes_through():
    """Handler returning _action_type=EXECUTE lets pipeline continue."""
    fast = Nucleus(provider=MockProvider(responses={}))
    deep = Nucleus(provider=MockProvider(responses={}))

    organism = skill_organism(
        stages=[
            SkillStage(name="ok", role="Echo",
                       handler=cli_handler("echo")),
            SkillStage(name="done", role="Done",
                       handler=lambda t: "finished"),
        ],
        fast_nucleus=fast,
        deep_nucleus=deep,
    )
    result = organism.run("hello")
    assert len(result.stage_results) == 2
    assert result.final_output == "finished"
