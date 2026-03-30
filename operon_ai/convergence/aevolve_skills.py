"""A-Evolve skill import bridge.

Imports A-Evolve workspace skills into an Operon :class:`PatternLibrary`
using the DeerFlow skill-to-template converter for Markdown content strings.

No A-Evolve imports; all inputs are plain strings/dicts.
"""

from __future__ import annotations

from typing import Any

from ..patterns.repository import PatternLibrary
from .aevolve_adapter import aevolve_to_template
from .deerflow_skills import skill_to_template


def import_aevolve_skills(skills: list[str], library: PatternLibrary) -> int:
    """Import A-Evolve SKILL.md content strings into a :class:`PatternLibrary`.

    Each string is a Markdown skill document (with optional YAML frontmatter
    and numbered workflow steps) that is converted via :func:`skill_to_template`
    from the DeerFlow skill bridge.

    Parameters
    ----------
    skills:
        List of Markdown content strings, one per skill file.
    library:
        Target :class:`PatternLibrary` to populate.

    Returns
    -------
    int
        Number of templates successfully registered (skips invalid skills).
    """
    count = 0
    for skill_md in skills:
        try:
            template = skill_to_template(skill_md)
            library.register_template(template)
            count += 1
        except (ValueError, KeyError):
            # Skip skills that cannot be parsed (e.g. no workflow steps).
            continue
    return count


def seed_library_from_aevolve(
    library: PatternLibrary,
    workspaces: list[dict[str, Any]],
) -> int:
    """Seed *library* with templates from A-Evolve workspace manifests.

    Parameters
    ----------
    library:
        Target :class:`PatternLibrary` to populate.
    workspaces:
        Each dict matches the shape expected by :func:`aevolve_to_template`.

    Returns
    -------
    int
        Number of templates successfully registered.
    """
    count = 0
    for manifest in workspaces:
        template = aevolve_to_template(manifest)
        library.register_template(template)
        count += 1
    return count
