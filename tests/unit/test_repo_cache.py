"""Tests for eval/_repo_cache.py.

All tests mock subprocess.run — no real network, no real git calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval._repo_cache import RepoCacheError, ensure_repo_at  # noqa: E402


def _ok(_cmd, **_kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _fail(cmd, **_kwargs):
    return SimpleNamespace(returncode=128, stdout="", stderr=f"fake failure for {cmd}")


def test_ensure_repo_at_first_call_runs_git(tmp_path):
    calls = []

    def track(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "init"]:
            # Simulate init creating the .git dir so the next call is a cache hit.
            (kwargs["cwd"] / ".git" if "cwd" not in kwargs
             else Path(kwargs["cwd"]) / ".git").mkdir(exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("eval._repo_cache.subprocess.run", side_effect=track):
        got = ensure_repo_at(
            "django/django", "abcdef1234567890" * 2, tmp_path,
        )
    assert got.parent == tmp_path
    assert got.name == "django__django-abcdef123456"
    # 4 subprocess calls: init, remote add, fetch, checkout.
    assert len(calls) == 4
    assert calls[0][:2] == ["git", "init"]
    assert calls[1][:3] == ["git", "remote", "add"]
    assert any(c[:2] == ["git", "fetch"] for c in calls)
    assert any(c[:2] == ["git", "checkout"] for c in calls)


def test_ensure_repo_at_caches_hit(tmp_path):
    # Pre-seed a "cloned" cache entry.
    key = "django__django-abcdef123456"
    target = tmp_path / key
    (target / ".git").mkdir(parents=True)

    with patch(
        "eval._repo_cache.subprocess.run", side_effect=AssertionError(
            "cache hit must not invoke git"
        ),
    ):
        got = ensure_repo_at(
            "django/django", "abcdef1234567890" * 2, tmp_path,
        )
    assert got == target


def test_ensure_repo_at_surfaces_git_failures_as_RepoCacheError(tmp_path):
    with patch("eval._repo_cache.subprocess.run", side_effect=_fail):
        with pytest.raises(RepoCacheError):
            ensure_repo_at(
                "django/django", "abcdef1234567890" * 2, tmp_path,
            )


def test_ensure_repo_at_cleans_up_partial_cache_on_failure(tmp_path):
    # First call fails after the directory is created but before .git exists.
    def partial(cmd, **_kwargs):
        # git init succeeds, everything else fails.
        if cmd[:2] == ["git", "init"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=128, stdout="", stderr="fetch failed")

    with patch("eval._repo_cache.subprocess.run", side_effect=partial):
        with pytest.raises(RepoCacheError):
            ensure_repo_at(
                "django/django", "abcdef1234567890" * 2, tmp_path,
            )

    # Half-populated dir is cleaned up so the next attempt starts fresh.
    assert not (tmp_path / "django__django-abcdef123456").exists()


def test_cache_key_separates_commits(tmp_path):
    # Two different commits of the same repo must get separate cache dirs.
    def track(_cmd, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # Pre-seed both entries so each call is a cache hit.
    a = tmp_path / "django__django-aaaaaaaaaaaa"
    b = tmp_path / "django__django-bbbbbbbbbbbb"
    (a / ".git").mkdir(parents=True)
    (b / ".git").mkdir(parents=True)

    with patch("eval._repo_cache.subprocess.run", side_effect=track):
        p1 = ensure_repo_at("django/django", "a" * 40, tmp_path)
        p2 = ensure_repo_at("django/django", "b" * 40, tmp_path)

    assert p1 == a
    assert p2 == b
    assert p1 != p2


def test_rejects_malformed_repo_slug(tmp_path):
    with pytest.raises(RepoCacheError, match="owner"):
        ensure_repo_at("no-slash", "a" * 40, tmp_path)


def test_rejects_short_commit(tmp_path):
    with pytest.raises(RepoCacheError, match="base_commit"):
        ensure_repo_at("django/django", "abc", tmp_path)
