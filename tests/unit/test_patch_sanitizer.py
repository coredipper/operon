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


def test_repairs_bare_empty_context_line():
    """A context line that lost its leading space (whitespace-stripped
    by an editor) must be repaired back to ``" "`` so git apply can
    consume it. Review #714.
    """
    # Body: 3 context (incl. one bare-empty) + 1 delete + 1 add.
    # Header counts both sides at 4 (3 context + 1 modify on each side).
    patch = (
        "--- a/django/foo.py\n"
        "+++ b/django/foo.py\n"
        "@@ -1,4 +1,4 @@\n"
        " context1\n"
        "\n"  # bare empty line — was " " before whitespace-stripping
        "-old\n"
        "+new\n"
        " context3\n"
    )
    cleaned = sanitize(patch, "django/django")
    assert cleaned, "bare empty context must be repaired, not rejected"
    # Count the two consecutive newlines that were there — after repair,
    # the empty line should have been rewritten to " " (a space + newline).
    assert "\n \n" in cleaned, (
        "bare empty line was not repaired to the required ' ' context "
        f"prefix; got:\n{cleaned!r}"
    )


def test_normalizes_rename_from_to_paths():
    """rename from/to metadata must get the same slug stripping as
    ---/+++ headers so the resulting diff is internally consistent.
    Review #714.
    """
    # django/django: the model doubled the repo name on all path-bearing
    # metadata. Sanitizer must strip consistently.
    patch = (
        "diff --git a/django/django/old.py b/django/django/new.py\n"
        "similarity index 95%\n"
        "rename from django/django/old.py\n"
        "rename to django/django/new.py\n"
        "--- a/django/django/old.py\n"
        "+++ b/django/django/new.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(patch, "django/django")
    assert cleaned, "valid rename diff must survive sanitize"
    # Every path-bearing line should be single-prefixed now.
    assert "rename from django/old.py" in cleaned
    assert "rename to django/new.py" in cleaned
    assert "diff --git a/django/old.py b/django/new.py" in cleaned
    assert "--- a/django/old.py" in cleaned
    assert "+++ b/django/new.py" in cleaned
    # And no trace of the doubled form anywhere.
    assert "django/django/" not in cleaned


def test_normalizes_copy_from_to_paths():
    """copy from/to metadata must be normalized just like rename."""
    patch = (
        "diff --git a/pallets/flask/src/a.py b/pallets/flask/src/b.py\n"
        "similarity index 100%\n"
        "copy from pallets/flask/src/a.py\n"
        "copy to pallets/flask/src/b.py\n"
        "--- a/pallets/flask/src/a.py\n"
        "+++ b/pallets/flask/src/b.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+y\n"
    )
    cleaned = sanitize(patch, "pallets/flask")
    assert cleaned
    assert "copy from src/a.py" in cleaned
    assert "copy to src/b.py" in cleaned
    assert "pallets/flask/" not in cleaned


# --------------------------------------------------------------------------
# Fuzzy path correction (Phase B — tree_paths oracle)
# --------------------------------------------------------------------------

def test_tree_paths_None_keeps_phase_A_behavior():
    """Omitting tree_paths must not change anything vs. Phase A."""
    # A doubled-path diff gets normalized (Phase A behavior) but no
    # fuzzy correction runs.
    doubled = (
        "--- a/django/django/foo.py\n"
        "+++ b/django/django/foo.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(doubled, "django/django")
    assert "a/django/foo.py" in cleaned


def test_fuzzy_rewrites_wrong_path_to_unique_basename_match():
    """Model named a nonexistent file; tree has exactly one file with
    that basename — rewrite to it."""
    patch = (
        "--- a/wrong/place/storage.py\n"
        "+++ b/wrong/place/storage.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    tree = frozenset({
        "django/core/files/storage.py",
        "django/forms/forms.py",
    })
    cleaned = sanitize(patch, "django/django", tree_paths=tree)
    assert cleaned
    assert "--- a/django/core/files/storage.py" in cleaned
    assert "+++ b/django/core/files/storage.py" in cleaned


def test_fuzzy_rejects_when_multiple_basename_matches():
    """Tree has two files with the same basename — can't guess, reject."""
    patch = (
        "--- a/wrong/place/utils.py\n"
        "+++ b/wrong/place/utils.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    tree = frozenset({
        "django/utils.py",
        "django/core/utils.py",
    })
    assert sanitize(patch, "django/django", tree_paths=tree) == ""


def test_fuzzy_rejects_when_no_basename_match():
    """Basename isn't in the tree at all — reject."""
    patch = (
        "--- a/invented.py\n"
        "+++ b/invented.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    tree = frozenset({"real.py", "other.py"})
    assert sanitize(patch, "django/django", tree_paths=tree) == ""


def test_fuzzy_leaves_correct_paths_untouched():
    """A diff whose paths are already in the tree must pass through
    unchanged."""
    patch = (
        "--- a/django/core/storage.py\n"
        "+++ b/django/core/storage.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    tree = frozenset({"django/core/storage.py"})
    cleaned = sanitize(patch, "django/django", tree_paths=tree)
    assert "--- a/django/core/storage.py" in cleaned


def test_fuzzy_accepts_dev_null_for_file_add_delete():
    """``/dev/null`` must always be accepted even when absent from the tree.

    A file-add diff has ``--- /dev/null`` and a real new path on the
    ``+++`` side. The oracle must let ``/dev/null`` through unchanged
    and only apply fuzzy correction to the real path.
    """
    tree = frozenset({"existing.py"})
    patch = (
        "diff --git a/existing.py b/existing.py\n"
        "--- /dev/null\n"
        "+++ b/existing.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+line\n"
    )
    assert sanitize(patch, "django/django", tree_paths=tree)


# --------------------------------------------------------------------------
# Review #724 follow-up: new-file / rename / copy targets must not be
# oracle-checked. They legitimately don't exist at base_commit.
# --------------------------------------------------------------------------

def test_fuzzy_accepts_new_file_target_even_when_absent_from_tree():
    """A create patch's target path is new by definition. Review #724."""
    tree = frozenset({"existing_a.py", "existing_b.py"})
    patch = (
        "diff --git a/brand_new.py b/brand_new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/brand_new.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+hello\n"
    )
    # brand_new.py is not in tree — but it's a CREATE, so it must not
    # be rewritten to "existing_a.py" or rejected.
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned, "create target should be accepted unchanged"
    assert "+++ b/brand_new.py" in cleaned
    # Must NOT have been rewritten to an unrelated unique basename match.
    assert "existing_a.py" not in cleaned


def test_fuzzy_rename_target_is_not_oracle_checked():
    """Rename target is a new path — must not be oracle-corrected.

    Source (rename from) must still be oracle-checked. Review #724.
    """
    tree = frozenset({"django/old_name.py", "django/unrelated.py"})
    patch = (
        "diff --git a/django/old_name.py b/django/new_name.py\n"
        "similarity index 95%\n"
        "rename from django/old_name.py\n"
        "rename to django/new_name.py\n"
        "--- a/django/old_name.py\n"
        "+++ b/django/new_name.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(patch, "django/django", tree_paths=tree)
    assert cleaned
    # Source paths (old_name.py) must have been verified against tree.
    assert "rename from django/old_name.py" in cleaned
    assert "--- a/django/old_name.py" in cleaned
    # Target paths (new_name.py) must be preserved verbatim even though
    # new_name.py is not in the tree.
    assert "rename to django/new_name.py" in cleaned
    assert "+++ b/django/new_name.py" in cleaned
    # And it must NOT have been rewritten to the unrelated file.
    assert "unrelated.py" not in cleaned


def test_fuzzy_copy_target_is_not_oracle_checked():
    """Copy target is a new path — same rule as rename. Review #724."""
    tree = frozenset({"src/source.py", "src/other.py"})
    patch = (
        "diff --git a/src/source.py b/src/dest.py\n"
        "similarity index 100%\n"
        "copy from src/source.py\n"
        "copy to src/dest.py\n"
        "--- a/src/source.py\n"
        "+++ b/src/dest.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(patch, "pallets/flask", tree_paths=tree)
    assert cleaned
    assert "copy from src/source.py" in cleaned
    assert "copy to src/dest.py" in cleaned
    assert "+++ b/src/dest.py" in cleaned
    # Not silently rewritten to src/other.py.
    assert "other.py" not in cleaned


def test_fuzzy_new_file_not_silently_rewritten_to_unrelated_match():
    """Review #724: the worst case is a new file's path being silently
    rewritten to some unrelated unique-basename match. Guard against
    that regression explicitly.
    """
    # Tree has ONE file named brand_new.py in a totally unrelated dir.
    # Without the fix, sanitize would rewrite the create's target to
    # that path. With the fix, the create target passes through unchanged.
    tree = frozenset({"totally/unrelated/brand_new.py"})
    patch = (
        "diff --git a/myapp/brand_new.py b/myapp/brand_new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/myapp/brand_new.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+content\n"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned
    assert "+++ b/myapp/brand_new.py" in cleaned
    assert "totally/unrelated" not in cleaned


def test_fuzzy_delete_source_must_exist():
    """A delete patch's source path must be in the tree (the file being
    deleted exists at base_commit). Target is /dev/null.
    """
    tree = frozenset({"django/to_remove.py"})
    patch = (
        "diff --git a/django/to_remove.py b/django/to_remove.py\n"
        "deleted file mode 100644\n"
        "--- a/django/to_remove.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-byebye\n"
    )
    cleaned = sanitize(patch, "django/django", tree_paths=tree)
    assert cleaned
    assert "--- a/django/to_remove.py" in cleaned
    assert "+++ /dev/null" in cleaned


def test_fuzzy_delete_source_absent_still_rejected():
    """Delete source must exist — if it doesn't, reject."""
    tree = frozenset({"other_file.py"})
    patch = (
        "diff --git a/invented.py b/invented.py\n"
        "deleted file mode 100644\n"
        "--- a/invented.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-x\n"
    )
    assert sanitize(patch, "django/django", tree_paths=tree) == ""


def test_fuzzy_modify_mirrors_source_correction_to_target():
    """In a plain modify, +++ b/<X> must match --- a/<X>. If we corrected
    the source, we must mirror the same correction on the target so the
    patch stays internally consistent.
    """
    tree = frozenset({"django/core/files/storage.py"})
    patch = (
        "diff --git a/wrong/storage.py b/wrong/storage.py\n"
        "--- a/wrong/storage.py\n"
        "+++ b/wrong/storage.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+b\n"
    )
    cleaned = sanitize(patch, "django/django", tree_paths=tree)
    assert cleaned
    assert "--- a/django/core/files/storage.py" in cleaned
    assert "+++ b/django/core/files/storage.py" in cleaned
    # diff --git line also mirrors the correction on both sides.
    assert "diff --git a/django/core/files/storage.py b/django/core/files/storage.py" in cleaned
