"""Interface for workflow generation from natural language (Phase C7)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..patterns.repository import PatternLibrary, PatternTemplate


@runtime_checkable
class WorkflowGenerator(Protocol):
    """Generate a PatternTemplate from a natural-language task description."""

    def generate(
        self,
        task_description: str,
        constraints: dict[str, Any] | None = None,
    ) -> PatternTemplate: ...

    def refine(
        self,
        template: PatternTemplate,
        feedback: str,
    ) -> PatternTemplate: ...


@runtime_checkable
class ReasoningGenerator(Protocol):
    """Extended protocol: returns reasoning trace alongside the template."""

    def reason_and_generate(
        self,
        task_description: str,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[PatternTemplate, list[str]]: ...


class HeuristicGenerator:
    """Reference implementation using default_template_generator."""

    def generate(
        self,
        task_description: str,
        constraints: dict[str, Any] | None = None,
    ) -> PatternTemplate:
        from .hybrid_assembly import default_template_generator

        return default_template_generator(task_description)

    def refine(
        self,
        template: PatternTemplate,
        feedback: str,
    ) -> PatternTemplate:
        return template


def generate_and_register(
    generator: WorkflowGenerator,
    task: str,
    library: PatternLibrary,
    constraints: dict[str, Any] | None = None,
) -> PatternTemplate:
    """Generate a template and register it in the library in one step."""
    template = generator.generate(task, constraints)
    library.register_template(template)
    return template
