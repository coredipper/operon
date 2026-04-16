"""Tests for eval/_repo_grounding.py.

Uses ``tmp_path`` to build synthetic repo trees — no real clones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval._repo_grounding import (  # noqa: E402
    extract_hints,
    format_context_block,
    rank_candidate_files,
    walk_tree_paths,
)


def _build_repo(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


# --------------------------------------------------------------------------
# extract_hints
# --------------------------------------------------------------------------

def test_extract_hints_picks_file_paths():
    hints = extract_hints("Storage.save() raises in django/core/storage.py.")
    assert "django/core/storage.py" in hints
    assert "storage" in hints  # stem


def test_extract_hints_picks_camel_case():
    hints = extract_hints("Storage.save() raises a FileSystemError")
    assert "Storage" in hints
    assert "FileSystemError" in hints


def test_extract_hints_picks_snake_case():
    hints = extract_hints("get_file_contents returns wrong value for empty_arg")
    assert "get_file_contents" in hints
    assert "empty_arg" in hints


def test_extract_hints_drops_stopwords():
    hints = extract_hints("The issue is a bug in this file.")
    # Stopwords lowercased in our filter — CamelCase "The" would be
    # filtered because "the" is a stopword.
    assert "The" not in hints
    assert "issue" not in hints


def test_extract_hints_empty_input():
    assert extract_hints("") == []
    assert extract_hints("   ") == []


def test_extract_hints_preserves_order_and_dedups():
    hints = extract_hints("foo_bar foo_bar Another Another")
    assert hints == ["Another", "foo_bar"]


# --------------------------------------------------------------------------
# rank_candidate_files
# --------------------------------------------------------------------------

def test_rank_prefers_stem_match_over_path_match(tmp_path):
    repo = _build_repo(tmp_path, {
        "django/core/storage.py": "class Storage: pass\n",
        "tests/helpers/storage_helper.py": "# unrelated\n",
        "docs/storage/intro.py": "# irrelevant doc\n",
    })
    ranked = rank_candidate_files(repo, ["storage"], k=5)
    # storage.py has stem match (+3) + path match (+1) = 4
    # storage_helper.py has path match only (+1)
    # docs/storage/intro.py has path match (+1)
    assert ranked[0].name == "storage.py"


def test_rank_tie_breaks_by_shorter_path(tmp_path):
    repo = _build_repo(tmp_path, {
        "a/foo.py": "x",
        "a/b/c/d/foo.py": "x",
    })
    ranked = rank_candidate_files(repo, ["foo"], k=5)
    assert ranked[0].name == "foo.py"
    assert str(ranked[0].relative_to(repo)) == "a/foo.py"


def test_rank_returns_empty_with_no_hints(tmp_path):
    repo = _build_repo(tmp_path, {"a.py": ""})
    assert rank_candidate_files(repo, [], k=5) == []


def test_rank_ignores_zero_score_files(tmp_path):
    repo = _build_repo(tmp_path, {
        "foo.py": "",
        "bar.py": "",
    })
    # "baz" matches nothing, so both files score 0.
    assert rank_candidate_files(repo, ["baz"], k=5) == []


def test_rank_skips_vcs_and_caches(tmp_path):
    repo = _build_repo(tmp_path, {
        ".git/config": "[core]",
        "__pycache__/foo.cpython-311.pyc": "binary",
        "src/foo.py": "",
    })
    ranked = rank_candidate_files(repo, ["foo"], k=5)
    assert len(ranked) == 1
    assert ranked[0].name == "foo.py"


# --------------------------------------------------------------------------
# format_context_block
# --------------------------------------------------------------------------

def test_format_context_block_includes_tree_and_snippets(tmp_path):
    repo = _build_repo(tmp_path, {
        "django/core/storage.py": "class Storage:\n    pass\n",
        "tests/foo.py": "# test",
    })
    ranked = rank_candidate_files(repo, ["Storage"], k=5)
    block = format_context_block(repo, ranked)
    assert "Top-level directories:" in block
    assert "django/" in block
    assert "tests/" in block
    assert "## django/core/storage.py" in block
    assert "class Storage:" in block


def test_format_context_block_truncates_long_files(tmp_path):
    big = "\n".join(f"line{i}" for i in range(500))
    repo = _build_repo(tmp_path, {"src/huge.py": big})
    ranked = rank_candidate_files(repo, ["huge"], k=5)
    block = format_context_block(repo, ranked, max_lines_per_file=10)
    assert "line0" in block
    assert "line9" in block
    assert "line499" not in block
    assert "truncated" in block


def test_format_context_block_empty_when_no_candidates(tmp_path):
    assert format_context_block(tmp_path, [], max_lines_per_file=10) == ""


def test_format_context_block_escapes_triple_backticks_in_file(tmp_path):
    """A file whose content contains ``` must not close the outer fence
    early — that would leak the rest of the file into the prompt as
    plain instructions (prompt-injection surface). Review #724.

    The outer fence is chosen to exceed any backtick run in the content.
    """
    # File content includes its own triple-backtick fenced block (as
    # might appear in a docstring markdown example).
    evil = "\n".join([
        "def foo():",
        '    """',
        "    Example:",
        "    ```python",
        "    foo()",
        "    ```",
        '    """',
        "    pass",
    ])
    repo = _build_repo(tmp_path, {"src/foo.py": evil})
    cands = rank_candidate_files(repo, ["foo"], k=5)
    block = format_context_block(repo, cands)

    # The outer fence must be at least 4 backticks (since content has 3).
    assert "````python" in block or "`````python" in block, (
        f"outer fence must be longer than 3 backticks when content has "
        f"```-runs; got:\n{block}"
    )
    # The full file content is preserved inside the outer fence.
    assert "def foo():" in block
    assert "foo()" in block


def test_format_context_block_handles_quadruple_backticks(tmp_path):
    """If a file contains even longer runs, the outer fence adapts."""
    evil = "```````` quadruple-backtick run ````````"
    repo = _build_repo(tmp_path, {"x.py": evil})
    cands = rank_candidate_files(repo, ["x"], k=5)
    block = format_context_block(repo, cands)
    # Outer fence must have at least 9 backticks (content has 8).
    assert "`" * 9 in block


# --------------------------------------------------------------------------
# walk_tree_paths
# --------------------------------------------------------------------------

def test_walk_tree_paths_collects_relative_paths(tmp_path):
    repo = _build_repo(tmp_path, {
        "a.py": "",
        "src/b.py": "",
        ".git/config": "",
        "__pycache__/c.pyc": "",
    })
    paths = walk_tree_paths(repo)
    assert "a.py" in paths
    assert "src/b.py" in paths
    # .git and __pycache__ must be skipped so the oracle doesn't suggest
    # suspicious targets.
    assert all(not p.startswith(".git/") for p in paths)
    assert all("__pycache__" not in p for p in paths)
