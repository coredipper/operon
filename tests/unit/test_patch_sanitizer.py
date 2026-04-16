"""Tests for eval/_patch_sanitizer.py.

Covers the three apply-failure modes this sanitizer eliminates before
the harness invokes git apply:

1. Paths doubled with the repo name (a/django/django/x.py)
2. Placeholder hunk headers (@@ -XXX,N +XXX,N @@)
3. Truncated hunks whose body doesn't match declared counts
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval._patch_sanitizer import sanitize  # noqa: E402


VALID_BASELINE = (
    "--- a/django/foo.py\n"
    "+++ b/django/foo.py\n"
    "@@ -1,2 +1,3 @@\n"
    " context\n"
    "-old\n"
    "+new\n"
    "+extra\n"
)


def test_empty_string_passes_through():
    assert sanitize("", "django/django") == ""
    assert sanitize("   ", "django/django") == ""


def test_noop_on_already_clean_diff():
    # Idempotence: sanitizing a clean diff doesn't change it.
    first = sanitize(VALID_BASELINE, "django/django")
    second = sanitize(first, "django/django")
    assert first == second
    assert first.startswith("--- a/django/foo.py\n")
    assert "@@ -1,2 +1,3 @@" in first


def test_strips_double_owner_prefix():
    doubled = (
        "--- a/django/django/foo.py\n"
        "+++ b/django/django/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " context\n"
        "-old\n"
        "+new\n"
    )
    cleaned = sanitize(doubled, "django/django")
    assert "--- a/django/foo.py" in cleaned
    assert "+++ b/django/foo.py" in cleaned
    # And the doubled form must be gone.
    assert "a/django/django/foo.py" not in cleaned


def test_strips_double_owner_prefix_different_owner_repo():
    doubled = (
        "diff --git a/pallets/flask/src/x.py b/pallets/flask/src/x.py\n"
        "--- a/pallets/flask/src/x.py\n"
        "+++ b/pallets/flask/src/x.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(doubled, "pallets/flask")
    assert "--- a/src/x.py" in cleaned
    assert "+++ b/src/x.py" in cleaned
    assert "diff --git a/src/x.py b/src/x.py" in cleaned


def test_strips_only_when_slug_matches():
    # If the path prefix is some other directory that happens to share
    # a name, don't strip.
    patch = (
        "--- a/other/django/foo.py\n"
        "+++ b/other/django/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(patch, "django/django")
    assert "a/other/django/foo.py" in cleaned


def test_rejects_placeholder_hunk_XXX():
    bad = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -XXX,10 +XXX,10 @@\n"
        "-old\n"
        "+new\n"
    )
    assert sanitize(bad, "django/django") == ""


def test_rejects_placeholder_hunk_single_letter():
    bad = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -N,10 +N,10 @@\n"
        "-old\n"
        "+new\n"
    )
    assert sanitize(bad, "django/django") == ""


def test_rejects_placeholder_hunk_question_mark():
    bad = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -?,? +?,? @@\n"
        "-old\n"
        "+new\n"
    )
    assert sanitize(bad, "django/django") == ""


def test_accepts_valid_hunk_header():
    assert sanitize(VALID_BASELINE, "django/django") != ""


def test_rejects_truncated_hunk_missing_additions():
    # Declared +1,5 but only 3 additions and 0 context in the body.
    bad = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,5 +1,5 @@\n"
        " ctx1\n"
        " ctx2\n"
        "-remove\n"
        "-remove2\n"
        "+add\n"
    )
    assert sanitize(bad, "django/django") == ""


def test_rejects_truncated_hunk_missing_deletions():
    # Declared -1,5 but only 1 deletion and 1 context.
    bad = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,5 +1,5 @@\n"
        " ctx\n"
        "-remove\n"
        "+add1\n"
        "+add2\n"
        "+add3\n"
    )
    assert sanitize(bad, "django/django") == ""


def test_accepts_no_newline_at_end_marker():
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )
    assert sanitize(patch, "django/django") != ""


def test_accepts_git_default_count_of_one():
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -42 +42 @@\n"
        "-old\n"
        "+new\n"
    )
    assert sanitize(patch, "django/django") != ""


def test_handles_multi_file_diff():
    patch = (
        "--- a/django/django/a.py\n"
        "+++ b/django/django/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-aa\n"
        "+AA\n"
        "--- a/django/django/b.py\n"
        "+++ b/django/django/b.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-bb\n"
        "+BB\n"
    )
    cleaned = sanitize(patch, "django/django")
    assert "--- a/django/a.py" in cleaned
    assert "--- a/django/b.py" in cleaned
    assert "django/django/" not in cleaned


def test_rejects_diff_with_no_hunks():
    # Headers only, no @@ body. git apply cannot use this.
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
    )
    assert sanitize(patch, "django/django") == ""


def test_trailing_newline_normalized():
    # Missing trailing newline gets added.
    patch_no_trailing = VALID_BASELINE.rstrip("\n")
    cleaned = sanitize(patch_no_trailing, "django/django")
    assert cleaned.endswith("\n")
    assert cleaned.count("\n\n") == 0 or cleaned.endswith("\n")
