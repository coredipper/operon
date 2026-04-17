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


# --------------------------------------------------------------------------
# Review #726: bare multi-file diffs (no ``diff --git`` headers) must
# correctly split into one block per file when grounding is active.
# --------------------------------------------------------------------------

def test_fuzzy_handles_bare_multi_file_diff_each_file_oracle_checked():
    """Bare diff with two modify sections back-to-back.

    Before the fix, both files collapsed into one block and the
    source_rewrite for the first leaked into the second.
    """
    tree = frozenset({"pkg/a.py", "pkg/b.py"})
    patch = (
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-aa\n"
        "+AA\n"
        "--- a/pkg/b.py\n"
        "+++ b/pkg/b.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-bb\n"
        "+BB\n"
    )
    cleaned = sanitize(patch, "owner/pkg", tree_paths=tree)
    assert cleaned
    assert "--- a/pkg/a.py" in cleaned
    assert "+++ b/pkg/a.py" in cleaned
    assert "--- a/pkg/b.py" in cleaned
    assert "+++ b/pkg/b.py" in cleaned


def test_fuzzy_bare_multi_file_mixed_modify_and_create():
    """Mixed: one modify + one create in a single bare diff. The create's
    target path must NOT be oracle-checked.
    """
    tree = frozenset({"pkg/existing.py"})
    patch = (
        "--- a/pkg/existing.py\n"
        "+++ b/pkg/existing.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "--- /dev/null\n"
        "+++ b/pkg/brand_new.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+hello\n"
    )
    cleaned = sanitize(patch, "owner/pkg", tree_paths=tree)
    assert cleaned, "mixed modify+create must survive"
    # Modify block intact.
    assert "--- a/pkg/existing.py" in cleaned
    assert "+++ b/pkg/existing.py" in cleaned
    # Create block preserved; brand_new.py NOT rewritten to existing.py
    # or rejected.
    assert "--- /dev/null" in cleaned
    assert "+++ b/pkg/brand_new.py" in cleaned


def test_fuzzy_bare_multi_file_mixed_with_bad_source_in_second():
    """Second file's source path is wrong and has a unique basename
    match — sanitizer rewrites the second without affecting the first.
    """
    tree = frozenset({
        "pkg/a.py",
        "pkg/deep/b.py",
    })
    patch = (
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+X\n"
        "--- a/wrong/b.py\n"
        "+++ b/wrong/b.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-y\n"
        "+Y\n"
    )
    cleaned = sanitize(patch, "owner/pkg", tree_paths=tree)
    assert cleaned
    # First block unchanged.
    assert "--- a/pkg/a.py" in cleaned
    assert "+++ b/pkg/a.py" in cleaned
    # Second block rewritten via unique basename match — mirrored on both
    # --- and +++.
    assert "--- a/pkg/deep/b.py" in cleaned
    assert "+++ b/pkg/deep/b.py" in cleaned
    assert "wrong/b.py" not in cleaned


def test_fuzzy_bare_multi_file_rejects_if_any_source_unresolvable():
    """If the second file's source can't be resolved, reject the whole
    patch — not just the second block.
    """
    tree = frozenset({"pkg/a.py"})
    patch = (
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-x\n"
        "+X\n"
        "--- a/invented.py\n"
        "+++ b/invented.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-y\n"
        "+Y\n"
    )
    assert sanitize(patch, "owner/pkg", tree_paths=tree) == ""


# --------------------------------------------------------------------------
# Review #729: a hunk-body deletion of content starting with "-- " is
# encoded as "--- ..." — must NOT be split as a file boundary. The next
# line being "+++ " is the disambiguator.
# --------------------------------------------------------------------------

def test_fuzzy_does_not_split_on_hunk_body_that_starts_with_dashes():
    """Hunk deletes a line whose content happens to start with ``-- a/``.

    Encoded as ``--- a/foo.py`` on the wire (``-`` delete marker +
    ``-- a/foo.py`` content). The next line is another body line, not
    ``+++ ``, so the sanitizer must keep it inside the hunk body and
    not treat it as a new file boundary.
    """
    # Single-file diff, but the hunk deletes two lines; one of the
    # deleted lines is literally ``-- a/foo.py``.
    tree = frozenset({"docs/readme.md"})
    patch = (
        "--- a/docs/readme.md\n"
        "+++ b/docs/readme.md\n"
        "@@ -1,3 +1,1 @@\n"
        " intro\n"
        "--- a/foo.py\n"
        "-next line\n"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned, (
        "hunk-body deletion starting with '-- ' must not split the "
        "file-diff; the whole patch should pass as a single-file "
        "modify"
    )
    # The deletion body is preserved verbatim inside the hunk.
    assert "--- a/foo.py" in cleaned  # still there, but as body, not header
    assert "-next line" in cleaned
    # And we did NOT produce a second block claiming foo.py is a source.
    # (If it had been misparsed as a header, the sanitizer would have
    # oracle-checked foo.py which is not in tree, so the whole patch
    # would have been rejected.)


def test_fuzzy_yaml_frontmatter_deletion_not_a_split():
    """Real-world case: deleting a YAML frontmatter block starts with
    ``---`` on each side, so a bare ``---`` deletion in the hunk body
    must not be split.
    """
    tree = frozenset({"docs/post.md"})
    patch = (
        "--- a/docs/post.md\n"
        "+++ b/docs/post.md\n"
        "@@ -1,4 +1,1 @@\n"
        "----\n"
        "-title: hello\n"
        "----\n"
        " content\n"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned
    # The YAML fence deletions (``----``) are preserved.
    assert "----" in cleaned


# --------------------------------------------------------------------------
# Review #730: hunk body with '--- a/...' delete followed by '+++ ...'
# add (arbitrary content starting with '++') still must NOT be split.
# Requires the sibling-line check to validate a real '+++' header
# shape, not just the '+++ ' prefix.
# --------------------------------------------------------------------------

def test_fuzzy_hunk_delete_dash_add_plus_not_split():
    """Hunk deletes content shaped ``-- a/foo.py`` (wire: ``--- a/foo.py``)
    AND adds content shaped ``++ plus`` (wire: ``+++ plus``). Before the
    fix, the lookahead would see ``+++ `` on the next line and misread
    the pair as a file header. The stricter check requires the next
    line to match ``+++ b/<path>`` or ``+++ /dev/null``.
    """
    # Body: 1 delete + 1 add + 2 context = 3 old, 3 new.
    tree = frozenset({"docs/readme.md"})
    patch = (
        "--- a/docs/readme.md\n"
        "+++ b/docs/readme.md\n"
        "@@ -1,3 +1,3 @@\n"
        "--- a/foo.py\n"        # body deletion of content "-- a/foo.py"
        "+++ some content\n"    # body addition of content "++ some content"
        " context\n"
        " context2\n"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned, (
        "hunk-body ambiguous dash/plus pair must not split the "
        "file-diff; got empty result"
    )
    assert "--- a/foo.py" in cleaned
    assert "+++ some content" in cleaned


# --------------------------------------------------------------------------
# Review #733: shape-only checks still match content shaped like
# '+++ b/...'. Count-driven hunk consumption is the real fix — inside a
# hunk body, the line's first character decides its role regardless of
# what follows.
# --------------------------------------------------------------------------

def test_fuzzy_hunk_body_with_plus_b_shape_not_split():
    """Hunk adds a line whose content starts with ``++ b/`` (wire:
    ``+++ b/foo.py``). A shape-only check would classify this as a
    real ``+++`` file header. Count-driven consumption recognizes it
    as an addition (starts with ``+``).
    """
    tree = frozenset({"docs/readme.md"})
    patch = (
        "--- a/docs/readme.md\n"
        "+++ b/docs/readme.md\n"
        "@@ -1,1 +1,2 @@\n"
        " context\n"
        "+++ b/bar.py\n"  # body add whose content is "++ b/bar.py"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned, "plus-shape hunk body must not be mis-split"
    assert "+++ b/bar.py" in cleaned
    # And the patch still parses as a single-file modify on readme.md.
    assert "--- a/docs/readme.md" in cleaned


def test_fuzzy_hunk_body_with_minus_a_plus_b_pair_not_split():
    """The review #733 adversarial case: delete shaped ``-- a/foo.py``
    paired with add shaped ``++ b/bar.py`` (wire: ``--- a/foo.py``
    then ``+++ b/bar.py``). Both lines individually match the shape
    of a file-header pair. Count-driven consumption is immune.
    """
    tree = frozenset({"docs/readme.md"})
    patch = (
        "--- a/docs/readme.md\n"
        "+++ b/docs/readme.md\n"
        "@@ -1,1 +1,1 @@\n"
        "--- a/foo.py\n"       # body delete
        "+++ b/bar.py\n"       # body add
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned, (
        "ambiguous --- / +++ body pair must not split the file-diff"
    )
    # Body content preserved verbatim.
    assert "--- a/foo.py" in cleaned
    assert "+++ b/bar.py" in cleaned
    # And the surviving patch is a single-file modify on readme.md —
    # NOT reinterpreted as a modify on foo.py → bar.py rename.
    # (If mis-split, the second block's a-side "foo.py" would have
    # been oracle-checked, not in tree, and the whole patch rejected.)


def test_sanitize_rejects_overlong_hunk():
    """Declared count ``@@ -1 +1 @@`` but body has an extra ``+``
    line. git apply would reject; sanitizer must too. Review #735.
    """
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "+extra\n"
    )
    assert sanitize(patch, "owner/repo") == ""


def test_sanitize_rejects_overlong_hunk_with_grounding():
    """Same rejection applies in grounding mode."""
    tree = frozenset({"foo.py"})
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "+extra\n"
    )
    assert sanitize(patch, "owner/repo", tree_paths=tree) == ""


def test_sanitize_rejects_overlong_hunk_extra_context():
    """Overlong context line after counts drain must also reject."""
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        " extra context\n"
    )
    assert sanitize(patch, "owner/repo") == ""


def test_sanitize_rejects_overlong_with_plus_b_shape_after_counts():
    """Review #736: an extra ``+++ b/bar.py`` after counts drain is
    overlong body content, not a legal post-hunk boundary. Shape-only
    acceptance would have let this through.
    """
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "+++ b/bar.py\n"   # addition of content "++ b/bar.py" — overlong
    )
    assert sanitize(patch, "owner/repo") == ""


def test_sanitize_rejects_overlong_with_minus_a_shape_after_counts():
    """Extra ``--- a/foo.py`` after counts drain without a real ``+++``
    sibling + ``@@`` triplet is overlong body content."""
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "--- a/foo.py\n"   # deletion of "-- a/foo.py" — overlong, not
                            # a file boundary (no +++ b/... followed by @@)
    )
    assert sanitize(patch, "owner/repo") == ""


def test_sanitize_rejects_overlong_with_plus_devnull_after_counts():
    """Extra ``+++ /dev/null`` after counts drain is overlong."""
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "+++ /dev/null\n"  # addition of "++ /dev/null" — overlong
    )
    assert sanitize(patch, "owner/repo") == ""


def test_sanitize_accepts_bare_multi_file_triplet_boundary():
    """A legitimate bare multi-file boundary must still be accepted:
    ``---`` + ``+++`` + ``@@`` triplet after the previous hunk's
    counts drain.
    """
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "--- a/bar.py\n"
        "+++ b/bar.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )
    cleaned = sanitize(patch, "owner/repo")
    assert cleaned
    assert "--- a/foo.py" in cleaned
    assert "--- a/bar.py" in cleaned


def test_sanitize_rejects_overlong_where_only_minus_plus_pair_without_hunk():
    """Two-line pair ``--- a/bar.py`` + ``+++ b/bar.py`` after counts
    drain WITHOUT a following ``@@`` is not a real boundary — must
    reject as overlong body content."""
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "--- a/bar.py\n"
        "+++ b/bar.py\n"   # no @@ follows — overlong body, not a boundary
    )
    assert sanitize(patch, "owner/repo") == ""


def test_sanitize_accepts_no_newline_marker_at_hunk_end():
    """``\\ No newline at end of file`` after the body is a valid
    marker and must NOT be rejected as overlong body content.
    """
    patch = (
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "\\ No newline at end of file\n"
    )
    assert sanitize(patch, "owner/repo")


def test_fuzzy_hunk_with_dev_null_shaped_body_add():
    """An added line whose content is ``/dev/null`` (wire: ``+/dev/null``
    wouldn't match, but ``+++ /dev/null`` would). Body content shaped
    as ``+++ /dev/null`` (a 3-char ``/++`` added) would also match
    our regex. Verify count-driven consumption handles it.
    """
    tree = frozenset({"x.py"})
    patch = (
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -1,1 +1,2 @@\n"
        " ctx\n"
        "+++ /dev/null\n"   # body add whose content is "++ /dev/null"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned
    assert "+++ /dev/null" in cleaned  # preserved as body, not header


def test_repair_preserves_bare_empty_after_ambiguous_delete():
    """A hunk with an ambiguous ``--- a/...`` deletion followed by a
    whitespace-stripped blank context line must still get the blank
    line repaired — the repair pass must use the same lookahead
    disambiguation as validation and splitting. Review #730.
    """
    # Body: ctx + delete "-- a/foo.py" + blank context + ctx2.
    # old = 3 (1 ctx + 1 delete + 1 blank-as-ctx + 1... no wait)
    # Let me recount: ctx(1,1) + delete(2,1) + blank-as-ctx(3,2) + ctx2(4,3)
    # So old=4, new=3. Header -1,4 +1,3.
    tree = frozenset({"docs/readme.md"})
    patch = (
        "--- a/docs/readme.md\n"
        "+++ b/docs/readme.md\n"
        "@@ -1,4 +1,3 @@\n"
        " ctx\n"
        "--- a/foo.py\n"   # body deletion, not a file header
        "\n"                # whitespace-stripped blank context
        " ctx2\n"
    )
    cleaned = sanitize(patch, "owner/repo", tree_paths=tree)
    assert cleaned, "repair must work even after ambiguous --- deletion"
    # Blank line repaired to ' ' (appears between the delete and ctx2).
    assert "--- a/foo.py\n \n ctx2" in cleaned
