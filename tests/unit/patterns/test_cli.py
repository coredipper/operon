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


# ---------------------------------------------------------------------------
# Regression tests for Job 64 findings
# ---------------------------------------------------------------------------


def test_shell_true_with_arg_mode():
    """shell=True + input_mode='arg' must include the task in the command."""
    h = cli_handler("printf %s", shell=True, input_mode="arg")
    result = h("hello")
    assert result["output"] == "hello"
    assert result["_action_type"] == "EXECUTE"


def test_stdin_preserves_special_characters_when_opted_out():
    """sanitize_task=False preserves punctuation and newlines in stdin."""
    h = cli_handler("cat", input_mode="stdin", sanitize_task=False)
    task = "line1\nline2() {x:$y}!"
    result = h(task)
    assert result["output"] == task


def test_stdin_sanitizes_by_default():
    """Default sanitize_task=True strips metacharacters and newlines on stdin."""
    h = cli_handler("cat", input_mode="stdin")  # sanitize_task=True by default
    task = "hello\n$(world)!"
    result = h(task)
    # Metacharacters $()! and newlines should be stripped
    assert "$" not in result["output"]
    assert "(" not in result["output"]
    assert "!" not in result["output"]
    assert "\n" not in result["output"]
    assert "hello" in result["output"]
    assert "world" in result["output"]


def test_action_type_not_mutated_on_reuse():
    """_coerce_handler_output must not mutate the handler's returned dict."""
    h = cli_handler("false")  # returns _action_type=FAILURE
    result1 = h("")
    assert result1["_action_type"] == "FAILURE"
    # Running the same handler again should still see FAILURE
    result2 = h("")
    assert result2["_action_type"] == "FAILURE"


def test_coerce_handler_output_does_not_mutate_shared_dict():
    """_coerce_handler_output should not pop _action_type from the original dict."""
    from operon_ai.patterns.organism import _coerce_handler_output

    shared = {"output": "test", "_action_type": "FAILURE"}
    protein = _coerce_handler_output(shared, "stage1")
    assert protein.action_type == "FAILURE"
    # Original dict must be unchanged
    assert "_action_type" in shared
