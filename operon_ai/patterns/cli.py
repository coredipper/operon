"""CLI stage handler — shell out to external CLI tools as organism stages.

Wraps any CLI command (Claude Code, Copilot, ruff, custom scripts) as a
SkillStage handler. The watcher, convergence detection, and developmental
gating all work unchanged on CLI-backed stages.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CLIResult:
    """Structured output from a CLI stage execution."""

    stdout: str
    stderr: str
    returncode: int
    command: str
    latency_ms: float
    timed_out: bool = False


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------


def parse_json(stdout: str) -> Any:
    """Parse JSON from stdout."""
    return json.loads(stdout.strip())


def parse_lines(stdout: str) -> list[str]:
    """Split stdout into non-empty lines."""
    return [line for line in stdout.strip().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_SHELL_META = re.compile(r'[;&|`$(){}\\<>!\n]')


def _sanitize(task: str) -> str:
    """Strip shell metacharacters from task input."""
    return _SHELL_META.sub("", task)


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def cli_handler(
    command: str | list[str],
    *,
    timeout: float = 30.0,
    input_mode: str = "arg",
    parse_output: Callable[[str], Any] | None = None,
    success_codes: tuple[int, ...] = (0,),
    shell: bool = False,
    sanitize_task: bool = True,
) -> Callable[..., dict[str, Any]]:
    """Build a SkillStage handler that shells out to a CLI tool.

    Args:
        command: CLI command as string or list of args.
        timeout: Max seconds before killing the process.
        input_mode: How to pass the task to the command:
            "arg" — append task as final argument
            "stdin" — pipe task to stdin
            "none" — run command without task input
        parse_output: Optional callable to transform stdout into structured data.
        success_codes: Return codes that count as success (default: only 0).
        shell: If True, run via shell (required for pipes/redirects). Use carefully.
        sanitize_task: Strip shell metacharacters from task when shell=False.

    Returns:
        A handler function compatible with SkillStage.handler.
    """

    def handler(task: str) -> dict[str, Any]:
        # Build command
        if isinstance(command, str):
            cmd_parts = shlex.split(command)
        else:
            cmd_parts = list(command)

        task_input = task
        if sanitize_task and not shell:
            task_input = _sanitize(task)

        stdin_data = None
        if input_mode == "arg":
            cmd_parts.append(task_input)
        elif input_mode == "stdin":
            stdin_data = task_input
        elif input_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown input_mode: {input_mode!r}")

        cmd_str = " ".join(cmd_parts) if not shell else (command if isinstance(command, str) else " ".join(command))

        # Execute
        start = time.time()
        timed_out = False
        try:
            result = subprocess.run(
                cmd_parts if not shell else cmd_str,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_data,
                shell=shell,
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Command timed out after {timeout}s"
            returncode = -1
            timed_out = True

        latency_ms = (time.time() - start) * 1000

        cli_result = CLIResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            command=cmd_str,
            latency_ms=latency_ms,
            timed_out=timed_out,
        )

        # Parse output
        output = stdout.strip()
        if parse_output is not None and stdout:
            try:
                output = parse_output(stdout)
            except Exception:
                pass  # Fall back to raw stdout

        # Action type
        action_type = "EXECUTE" if returncode in success_codes else "FAILURE"

        return {
            "output": output,
            "cli_result": cli_result,
            "_action_type": action_type,
        }

    return handler


# ---------------------------------------------------------------------------
# Convenience: cli_organism
# ---------------------------------------------------------------------------


def cli_organism(
    commands: dict[str, str | list[str]],
    *,
    timeout: float = 30.0,
    input_mode: str = "arg",
    parse_output: Callable[[str], Any] | None = None,
    success_codes: tuple[int, ...] = (0,),
    **managed_kwargs: Any,
) -> Any:
    """Build a ManagedOrganism from a dict of CLI commands.

    Each key becomes a stage name, each value becomes a cli_handler command.

    Example:
        m = cli_organism({"generate": "echo", "lint": "true"})
        result = m.run("hello")
    """
    from .managed import managed_organism
    from .types import SkillStage

    stages = [
        SkillStage(
            name=name,
            role=name.title(),
            handler=cli_handler(
                cmd,
                timeout=timeout,
                input_mode=input_mode,
                parse_output=parse_output,
                success_codes=success_codes,
            ),
        )
        for name, cmd in commands.items()
    ]

    return managed_organism(stages=stages, **managed_kwargs)
