"""Sanitize extracted unified diffs before handing them to git apply.

This runs downstream of ``eval._patch_extraction.extract_patch`` and
eliminates three deterministic failure modes that otherwise show up as
opaque ``git apply`` errors in the SWE-bench harness:

1. Placeholder hunk headers (``@@ -XXX,10 +XXX,10 @@``). The model
   emitted a template rather than real line numbers. git apply rejects
   with ``"missing line number"``.
2. Path doubling (``a/django/django/x.py`` when the repo is
   ``django/django``). The model prefixed the repo name into the diff
   path. git apply rejects with ``"can't find file to patch"``.
3. Truncated hunks where declared ``@@ -a,b +c,d @@`` counts don't
   match the actual body. git apply rejects with
   ``"unexpected end of file in patch"``.

Returning ``""`` is a deliberate signal: the harness counts it as
``empty_patch`` rather than ``error``, which is honest — we refused to
submit something we knew would fail to apply.
"""

from __future__ import annotations

import re


# Patterns that appear inside diff headers but are NOT hunk body lines.
# These must be excluded when counting additions/deletions/context.
_METADATA_PREFIXES = (
    "diff --git ",
    "index ",
    "new file mode",
    "deleted file mode",
    "old mode",
    "new mode",
    "rename from",
    "rename to",
    "copy from",
    "copy to",
    "similarity index",
    "dissimilarity index",
    "Binary files ",
    "GIT binary patch",
)


_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>[^,\s]+)(?:,(?P<old_count>[^\s]+))? "
    r"\+(?P<new_start>[^,\s]+)(?:,(?P<new_count>[^\s]+))? @@"
)


def sanitize(patch: str, repo_slug: str) -> str:
    """Return a sanitized unified diff, or ``""`` if unsalvageable.

    ``repo_slug`` is the SWE-bench ``repo`` field, e.g. ``"django/django"``.
    An empty return means the caller should treat the submission as
    ``empty_patch`` rather than forward a broken diff to ``git apply``.
    """
    if not patch or not patch.strip():
        return ""

    normalized = _normalize_paths(patch, repo_slug)
    normalized = _repair_bare_empty_context(normalized)
    if not _validate_hunks(normalized):
        return ""

    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _repair_bare_empty_context(patch: str) -> str:
    """Restore the ``" "`` prefix on bare empty context lines inside hunks.

    Some editors/tools strip trailing whitespace, which turns a valid
    context line (a single space followed by a newline, representing a
    blank line in the file) into a bare empty line. ``git apply``
    rejects bare empty lines inside hunks, so we rewrite them back to
    ``" "``. Only applied inside hunk bodies — headers and separating
    whitespace between files are left alone.
    """
    lines = patch.splitlines()
    out: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            out.append(line)
            continue
        if in_hunk:
            if (line.startswith("--- ")
                    or line.startswith("+++ ")
                    or line.startswith(_METADATA_PREFIXES)):
                in_hunk = False
                out.append(line)
                continue
            if line == "":
                # Blank line inside a hunk = whitespace-stripped context.
                out.append(" ")
                continue
        out.append(line)
    return "\n".join(out)


def _normalize_paths(patch: str, repo_slug: str) -> str:
    """Strip duplicated ``{owner}/{repo}/`` prefix from file headers.

    The normalization is owner/repo-name-dependent because the two
    common cases have different ground-truth layouts:

    * Repos where ``owner == repo_name`` (e.g. ``django/django``,
      ``astropy/astropy``): there IS a top-level package directory
      named after the repo, so ``django/forms/x.py`` is a legitimate
      path. The wrong form is ``django/django/forms/x.py`` (the model
      doubled the segment); the fix is to strip ONE ``django/``,
      leaving ``django/forms/x.py``. A single-prefixed path is left
      alone.

    * Repos where ``owner != repo_name`` (e.g. ``pallets/flask``): the
      owner does NOT appear in the real tree. The wrong form is
      ``pallets/flask/src/x.py`` and the fix is to strip the whole
      ``pallets/flask/`` prefix, leaving ``src/x.py``.
    """
    owner, _, repo_name = repo_slug.partition("/")
    if not owner or not repo_name:
        return patch

    owner_repo_prefix = f"{owner}/{repo_name}/"
    owner_prefix = f"{owner}/"

    def _strip_side(path: str) -> str:
        # path is everything after "a/" or "b/".
        if owner == repo_name:
            # Only strip if we see the doubled form. Leave legitimate
            # single-prefix paths (``django/forms/x.py``) untouched.
            if path.startswith(owner_repo_prefix):
                return path[len(owner_prefix):]
            return path
        else:
            if path.startswith(owner_repo_prefix):
                return path[len(owner_repo_prefix):]
            return path

    out = []
    for line in patch.splitlines():
        stripped = _strip_diff_line_prefix(line, _strip_side)
        out.append(stripped)
    return "\n".join(out)


# Git metadata lines whose value is a bare path (no ``a/``/``b/`` prefix).
# These must be normalized alongside ``---``/``+++`` so a rename or copy
# diff does not end up with inconsistent old vs. new paths after the
# slug prefix is stripped from one side but not the other.
_BARE_PATH_PREFIXES = (
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)


def _strip_diff_line_prefix(line: str, strip_side) -> str:
    """Rewrite a single diff header line's path if applicable.

    Handles ``--- a/...``, ``+++ b/...``, ``diff --git a/... b/...``,
    and the bare-path metadata forms ``rename from``, ``rename to``,
    ``copy from``, ``copy to``. Leaves ``/dev/null`` untouched (file
    add/delete marker).
    """
    if line.startswith("--- a/"):
        tail = line[len("--- a/"):]
        return "--- a/" + strip_side(tail)
    if line.startswith("+++ b/"):
        tail = line[len("+++ b/"):]
        return "+++ b/" + strip_side(tail)
    if line.startswith("diff --git "):
        # form: diff --git a/<path> b/<path>
        m = re.match(
            r"^(diff --git )a/(\S+) b/(\S+)(.*)$", line
        )
        if m:
            prefix, a_path, b_path, rest = m.groups()
            return f"{prefix}a/{strip_side(a_path)} b/{strip_side(b_path)}{rest}"
    for meta in _BARE_PATH_PREFIXES:
        if line.startswith(meta):
            tail = line[len(meta):]
            return meta + strip_side(tail)
    return line


def _validate_hunks(patch: str) -> bool:
    """Return True if every hunk's declared counts match its body.

    Rejects placeholder line numbers (non-digit ``XXX``/``N``/``?``) and
    truncated hunks where the body line counts disagree with the header.
    """
    lines = patch.splitlines()
    i = 0
    saw_any_hunk = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            saw_any_hunk = True
            m = _HUNK_RE.match(line)
            if not m:
                return False
            # All of the numeric fields must be actual base-10 integers.
            # Reject placeholders like XXX, N, Y, single letters, ?.
            for key in ("old_start", "old_count", "new_start", "new_count"):
                val = m.group(key)
                if val is None:
                    # Omitted count is valid — git default is 1.
                    continue
                if not val.isdigit():
                    return False
            old_count = int(m.group("old_count")) if m.group("old_count") else 1
            new_count = int(m.group("new_count")) if m.group("new_count") else 1

            # Walk the body until the next hunk or the next file header.
            body_old = 0
            body_new = 0
            i += 1
            while i < len(lines):
                body_line = lines[i]
                if body_line.startswith("@@"):
                    break
                if (body_line.startswith("--- ")
                        or body_line.startswith("+++ ")
                        or body_line.startswith(_METADATA_PREFIXES)):
                    break
                if body_line == r"\ No newline at end of file":
                    # Marker, not a real line — ignored by both sides.
                    i += 1
                    continue
                if body_line.startswith("+"):
                    body_new += 1
                elif body_line.startswith("-"):
                    body_old += 1
                elif body_line.startswith(" "):
                    body_old += 1
                    body_new += 1
                else:
                    # Bare empty lines are repaired upstream by
                    # _repair_bare_empty_context; anything else here is
                    # malformed.
                    return False
                i += 1
            if body_old != old_count or body_new != new_count:
                return False
            continue  # already advanced i
        i += 1

    # An input with no hunks at all is not a valid unified diff to submit.
    return saw_any_hunk
