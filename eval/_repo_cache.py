"""Shallow-clone a GitHub repo at a specific commit for SWE-bench grounding.

The SWE-bench harness already clones inside its own Docker containers for
evaluation. This module is separate: it gives the *prompt-building* phase
enough of the target repo (at ``base_commit``) to inject real file paths
and file contents into the model's context before the patch is generated.

Cache layout (one directory per unique (repo, commit) pair):

    {cache_dir}/
        {owner}__{repo}-{commit[:12]}/
            .git/
            <repo tree at base_commit>

Each cache entry is created with::

    git init
    git remote add origin https://github.com/{repo_slug}.git
    git fetch --depth 1 origin {base_commit}
    git checkout FETCH_HEAD

No new dependencies — stdlib ``subprocess`` only. The caller supplies
the ``cache_dir`` and decides what to do on failure via
``RepoCacheError``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class RepoCacheError(RuntimeError):
    """Raised when shallow-cloning ``repo@commit`` cannot complete."""


def ensure_repo_at(
    repo_slug: str, base_commit: str, cache_dir: Path,
) -> Path:
    """Return the path to a checkout of ``repo_slug`` at ``base_commit``.

    If the cache entry already exists and looks valid (has a ``.git``
    directory), no git calls are made — the existing path is returned.
    Otherwise the repo is shallow-fetched at the specific commit and
    checked out.

    Cache keys include a 12-char commit prefix so different commits of
    the same repo don't collide.
    """
    if "/" not in repo_slug:
        raise RepoCacheError(
            f"repo_slug must be '<owner>/<repo>', got {repo_slug!r}"
        )
    if not base_commit or len(base_commit) < 7:
        raise RepoCacheError(
            f"base_commit looks invalid: {base_commit!r}"
        )
    owner, _, repo_name = repo_slug.partition("/")
    key = f"{owner}__{repo_name}-{base_commit[:12]}"
    target = cache_dir / key

    if (target / ".git").exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    try:
        _run(["git", "init", "--quiet"], target)
        _run(
            ["git", "remote", "add", "origin",
             f"https://github.com/{repo_slug}.git"],
            target,
        )
        _run(
            ["git", "fetch", "--quiet", "--depth", "1", "origin",
             base_commit],
            target,
        )
        _run(["git", "checkout", "--quiet", "FETCH_HEAD"], target)
    except RepoCacheError:
        # Clean up a half-populated cache entry so the next call starts
        # fresh rather than seeing an invalid state.
        _rm_tree(target)
        raise
    return target


def _run(cmd: list[str], cwd: Path) -> None:
    """Run ``cmd`` inside ``cwd``. Raise RepoCacheError on any failure."""
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd),
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RepoCacheError(f"{cmd[0]} failed: {e}") from e
    if result.returncode != 0:
        stderr = (result.stderr or "").strip() or "(no stderr)"
        raise RepoCacheError(
            f"{' '.join(cmd)} exited {result.returncode}: {stderr}"
        )


def _rm_tree(path: Path) -> None:
    """Best-effort recursive delete. Swallows errors so cleanup never
    masks the original exception."""
    if not path.exists():
        return
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
