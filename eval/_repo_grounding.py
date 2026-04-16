"""Build a repository-context block from issue text and a cloned repo.

Downstream of ``eval._repo_cache.ensure_repo_at``: given the checkout
and the issue text, produce a string that names plausible files and
shows their contents. That string is injected into the SWE-bench task
prompt so the model doesn't hallucinate paths or line numbers.

All operations are heuristic and read-only — no LLM calls, no network.
Gold-patch fields (``instance["patch"]``, ``instance["test_patch"]``)
are never touched; only the issue text the model also sees is mined
for hints.
"""

from __future__ import annotations

import re
from pathlib import Path


# Stopwords likely to appear in issue text but never useful as file-name
# hints. Kept deliberately short — aggressive filtering would drop real
# identifiers.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "of",
    "to", "for", "with", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "i", "you", "we",
    "not", "no", "so", "do", "does", "did", "has", "have", "had",
    "can", "could", "should", "would", "will", "may", "might", "must",
    "issue", "bug", "fix", "error", "please", "when", "what", "why",
    "how", "which", "problem", "expected", "actual", "result",
})


_FILE_PATH_RE = re.compile(r"\b([A-Za-z_][\w/]*\.py)\b")
_CAMEL_CASE_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]+\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b")


def extract_hints(text: str) -> list[str]:
    """Pull plausible file names, module paths, and identifiers from text.

    The returned list is deduplicated but order-preserving — callers can
    treat it as a priority order where earlier hints matter more.
    """
    if not text:
        return []
    seen: dict[str, None] = {}

    # Explicit ``.py`` paths carry the most signal.
    for m in _FILE_PATH_RE.findall(text):
        seen.setdefault(m, None)
        # Also add the filename without extension so ``foo.py`` mentioned
        # in the issue matches a file on disk.
        stem = m.rsplit("/", 1)[-1]
        if stem.endswith(".py"):
            seen.setdefault(stem[:-3], None)

    # CamelCase identifiers (class names, error types).
    for m in _CAMEL_CASE_RE.findall(text):
        if m.lower() in _STOPWORDS:
            continue
        seen.setdefault(m, None)

    # snake_case identifiers (function names, module names).
    for m in _SNAKE_CASE_RE.findall(text):
        if m in _STOPWORDS:
            continue
        seen.setdefault(m, None)

    return list(seen.keys())


def rank_candidate_files(
    repo_path: Path, hints: list[str], k: int = 5,
) -> list[Path]:
    """Rank Python files in ``repo_path`` by how well they match ``hints``.

    Scoring:
      - +3 if the file's stem (filename without .py) appears in hints
      - +1 for each hint appearing anywhere in the file's relative path
      - tie-break: shorter path first (top-of-tree usually more
        relevant than deep test fixtures)
    """
    if not hints or not repo_path.exists():
        return []
    hints_lower = {h.lower() for h in hints}
    scored: list[tuple[int, int, Path]] = []  # (score, path_length, path)
    for py in repo_path.rglob("*.py"):
        if _should_skip(py, repo_path):
            continue
        rel = py.relative_to(repo_path)
        rel_str = str(rel).lower()
        stem_lower = py.stem.lower()
        score = 0
        if stem_lower in hints_lower:
            score += 3
        for h in hints_lower:
            if h and h in rel_str:
                score += 1
        if score == 0:
            continue
        scored.append((score, len(str(rel)), py))
    scored.sort(key=lambda t: (-t[0], t[1], str(t[2])))
    return [p for _, _, p in scored[:k]]


def format_context_block(
    repo_path: Path, paths: list[Path], max_lines_per_file: int = 200,
) -> str:
    """Format candidate files as a prompt-ready context block.

    Shape::

        Top-level directories:
        - django/
        - tests/

        Candidate files (ranked):
        ## django/core/storage.py
        ```python
        <first N lines>
        ```
    """
    if not paths:
        return ""
    parts: list[str] = []
    top = _top_level_listing(repo_path)
    if top:
        parts.append("Top-level directories:")
        parts.extend(f"- {d}" for d in top)
        parts.append("")
    parts.append("Candidate files (ranked):")
    for p in paths:
        try:
            rel = p.relative_to(repo_path)
        except ValueError:
            continue
        content = _read_head(p, max_lines_per_file)
        parts.append(f"## {rel}")
        parts.append("```python")
        parts.append(content)
        parts.append("```")
        parts.append("")
    return "\n".join(parts)


def walk_tree_paths(repo_path: Path) -> frozenset[str]:
    """Return all repo-relative file paths as a frozenset.

    Used as the ``tree_paths`` oracle for sanitizer fuzzy path correction.
    """
    if not repo_path.exists():
        return frozenset()
    out = set()
    for p in repo_path.rglob("*"):
        if _should_skip(p, repo_path):
            continue
        if p.is_file():
            try:
                out.add(str(p.relative_to(repo_path)))
            except ValueError:
                pass
    return frozenset(out)


def _should_skip(path: Path, repo_path: Path) -> bool:
    """Skip version control, caches, venvs, etc."""
    try:
        rel = path.relative_to(repo_path)
    except ValueError:
        return True
    parts = rel.parts
    skip_dirs = {".git", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
                 ".pytest_cache", "node_modules", "build", "dist", ".eggs"}
    return any(p in skip_dirs for p in parts)


def _top_level_listing(repo_path: Path, limit: int = 25) -> list[str]:
    if not repo_path.exists():
        return []
    out = []
    for entry in sorted(repo_path.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            out.append(f"{entry.name}/")
        else:
            out.append(entry.name)
        if len(out) >= limit:
            break
    return out


def _read_head(path: Path, max_lines: int) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"# ... truncated ({max_lines} lines) ...")
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except OSError:
        return "# <unreadable>"
