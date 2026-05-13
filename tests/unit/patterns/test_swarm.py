"""Tests for the specialist swarm pattern."""

from unittest.mock import patch

import pytest

from operon_ai.patterns.swarm import (
    SpecialistSwarm,
    _call_aggregate,
    _default_aggregate,
    _default_worker,
    specialist_swarm,
)
from operon_ai.utils import call_arity
from operon_ai.patterns.types import SpecialistSwarmResult


# -- Initialization Tests --


def test_specialist_swarm_initialization():
    """Test successful creation of a specialist swarm."""
    swarm = specialist_swarm(roles=["analyst", "reviewer"])
    assert isinstance(swarm, SpecialistSwarm)
    assert swarm.config.roles == ("analyst", "reviewer")
    assert swarm.config.aggregation == "hub"
    assert swarm.analysis is not None
    assert swarm.diagram is not None
    assert "analyst" in swarm.diagram.modules
    assert "reviewer" in swarm.diagram.modules
    assert "coordinator" in swarm.diagram.modules


def test_specialist_swarm_empty_roles():
    """Test ValueError is raised when roles is empty."""
    with pytest.raises(ValueError, match="requires at least one role"):
        specialist_swarm(roles=[])


def test_specialist_swarm_duplicate_roles():
    """Test ValueError is raised when roles contain duplicates."""
    with pytest.raises(ValueError, match="roles must be unique"):
        specialist_swarm(roles=["analyst", "analyst"])


def test_specialist_swarm_invalid_aggregation():
    """Test ValueError is raised when aggregation is not 'hub'."""
    with pytest.raises(
        ValueError, match="currently supports aggregation='hub' only"
    ):
        specialist_swarm(roles=["analyst"], aggregation="tree")


# -- Helper Function Tests --


def test_call_arity():
    """Test `call_arity` correctly invokes functions with different signatures."""

    def no_args():
        return "no_args"

    def one_arg(prompt):
        return f"prompt={prompt}"

    def kwargs_arg(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    assert call_arity(no_args, "the_prompt", "the_role") == "no_args"
    assert call_arity(one_arg, "the_prompt", "the_role") == "prompt=the_prompt"
    assert call_arity(kwargs_arg, "the_prompt", "the_role") == {"args": ("the_prompt", "the_role"), "kwargs": {}}


def test_call_aggregate():
    """Test `_call_aggregate` correctly invokes functions with different signatures."""

    def zero_args():
        return "zero_args"

    def one_arg(outputs):
        return f"keys={list(outputs.keys())}"

    def two_args(task, outputs):
        return f"task={task}, keys={list(outputs.keys())}"

    def kwargs_arg(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    task_str = "test task"
    outputs_dict = {"a": 1, "b": 2}

    assert _call_aggregate(zero_args, task_str, outputs_dict) == "zero_args"
    assert _call_aggregate(one_arg, task_str, outputs_dict) == "keys=['a', 'b']"
    assert _call_aggregate(two_args, task_str, outputs_dict) == "task=test task, keys=['a', 'b']"
    assert _call_aggregate(kwargs_arg, task_str, outputs_dict) == {"args": ("test task", {"a": 1, "b": 2}), "kwargs": {}}


def test_call_arity_signature_error():
    """Test `call_arity` falls back correctly when signature inspection fails."""
    def mock_fn(*args):
        return args

    with patch("operon_ai.utils.function_utils.signature", side_effect=ValueError):
        assert call_arity(mock_fn, "prompt", "role") == ("prompt", "role")

    with patch("operon_ai.utils.function_utils.signature", side_effect=TypeError):
        assert call_arity(mock_fn, "prompt", "role") == ("prompt", "role")


def test_call_aggregate_signature_error():
    """Test `_call_aggregate` falls back correctly when signature inspection fails."""
    def mock_fn(*args):
        return args

    with patch("operon_ai.patterns.swarm.signature", side_effect=ValueError):
        assert _call_aggregate(mock_fn, "task", {"a": 1}) == ("task", {"a": 1})

    with patch("operon_ai.patterns.swarm.signature", side_effect=TypeError):
        assert _call_aggregate(mock_fn, "task", {"a": 1}) == ("task", {"a": 1})


def test_default_worker():
    """Test `_default_worker` behavior."""
    res = _default_worker("analyst", "some task")
    assert res == {
        "role": "analyst",
        "summary": "analyst reviewed the task",
        "task": "some task",
    }


def test_default_aggregate():
    """Test `_default_aggregate` behavior."""
    # Dict of strings
    res1 = _default_aggregate({"role1": "output1", "role2": "output2"})
    assert isinstance(res1, str)
    assert "[role1] output1" in res1
    assert "[role2] output2" in res1

    # Mixed or dict of non-strings
    res2 = _default_aggregate({"role1": {"nested": "value"}})
    assert res2 == {"role1": {"nested": "value"}}


# -- Execution Tests --


def test_specialist_swarm_run_default():
    """Test executing a swarm with default workers and aggregator."""
    swarm = specialist_swarm(roles=["worker_a", "worker_b"])
    result = swarm.run("test prompt")

    assert isinstance(result, SpecialistSwarmResult)

    # Check outputs
    assert "worker_a" in result.outputs
    assert "worker_b" in result.outputs

    assert result.outputs["worker_a"]["role"] == "worker_a"
    assert result.outputs["worker_a"]["task"] == "test prompt"

    assert result.outputs["worker_b"]["role"] == "worker_b"
    assert result.outputs["worker_b"]["task"] == "test prompt"

    # With non-string defaults, default_aggregate should just return the dict
    assert result.aggregate == result.outputs


def test_specialist_swarm_run_custom():
    """Test executing a swarm with custom workers and a custom aggregator."""

    def custom_worker_a(prompt):
        return f"A processing: {prompt}"

    def custom_worker_b(prompt, role):
        return f"{role} also processing: {prompt}"

    def custom_aggregator(outputs):
        return " | ".join(outputs.values())

    workers = {
        "worker_a": custom_worker_a,
        "worker_b": custom_worker_b,
    }

    swarm = specialist_swarm(
        roles=["worker_a", "worker_b"],
        workers=workers,
        aggregator=custom_aggregator,
    )
    result = swarm.run("do the work")

    assert result.outputs["worker_a"] == "A processing: do the work"
    assert result.outputs["worker_b"] == "worker_b also processing: do the work"

    # The aggregator joins string outputs
    assert result.aggregate == "A processing: do the work | worker_b also processing: do the work"
