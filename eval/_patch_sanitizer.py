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


def sanitize(
    patch: str, repo_slug: str,
    *, tree_paths: frozenset[str] | None = None,
) -> str:
    """Return a sanitized unified diff, or ``""`` if unsalvageable.

    ``repo_slug`` is the SWE-bench ``repo`` field, e.g. ``"django/django"``.
    An empty return means the caller should treat the submission as
    ``empty_patch`` rather than forward a broken diff to ``git apply``.

    ``tree_paths``, when provided, is the set of repo-relative file
    paths that actually exist at ``base_commit``. It is used as an
    oracle to fuzzy-correct paths: a diff pointing at a nonexistent
    file may be rewritten to a file with the same basename if the
    match is unique, or rejected if it's ambiguous or impossible. When
    ``tree_paths`` is ``None``, no oracle-based correction runs —
    behavior matches the Phase A sanitizer exactly.
    """
    if not patch or not patch.strip():
        return ""

    normalized = _normalize_paths(patch, repo_slug)
    normalized = _repair_bare_empty_context(normalized)
    if tree_paths is not None:
        normalized = _fuzzy_correct_paths(normalized, tree_paths)
        if not normalized:
            return ""
    if not _validate_hunks(normalized):
        return ""

    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _fuzzy_correct_paths(patch: str, tree_paths: frozenset[str]) -> str:
    """Rewrite diff paths so each file-diff's source paths exist in
    ``tree_paths``, while target paths of create / rename / copy
    operations are left alone (those paths legitimately don't exist
    at ``base_commit``).

    Per file-diff (a block starting at ``diff --git`` or ``---``):

    * Source-side paths must exist at base_commit. The source is
      ``--- a/<p>`` (for modify/delete/rename-from), ``rename from``,
      ``copy from``. If a source path is missing from the tree, try
      to rewrite it to a unique basename match; if that fails, reject
      the whole patch.
    * Target-side paths of a creation (``--- /dev/null``), rename
      (``rename to``), or copy (``copy to``) are passed through
      unchanged — they are new paths by definition.
    * For a plain modify, source == target, and the ``+++ b/<p>`` line
      is rewritten to mirror any correction applied to ``--- a/<p>``.
      The ``diff --git a/<X> b/<Y>`` line is rewritten the same way.
    * ``/dev/null`` is always accepted.

    Returns ``""`` when any source-side path cannot be resolved.
    """
    basename_index: dict[str, list[str]] = {}
    for p in tree_paths:
        base = p.rsplit("/", 1)[-1]
        basename_index.setdefault(base, []).append(p)

    def _resolve_source(path: str) -> str | None:
        """Oracle check for a source-side path."""
        if path == "/dev/null":
            return path
        if path in tree_paths:
            return path
        base = path.rsplit("/", 1)[-1]
        matches = basename_index.get(base, [])
        if len(matches) == 1:
            return matches[0]
        return None

    blocks = _split_file_diffs(patch)
    out_parts: list[str] = []
    for block in blocks:
        fixed = _correct_file_diff(block, _resolve_source)
        if fixed is None:
            return ""
        out_parts.append(fixed)
    return "\n".join(out_parts)


_FILE_HEADER_RE = re.compile(r"^--- (?:a/|/dev/null$|/dev/null\t)")


def _is_minus_file_header(line: str, next_line: str | None) -> bool:
    """Return True iff ``line`` is a real unified-diff ``---`` file
    header (not a hunk deletion that happens to start with ``-- ``).

    A deletion of content starting with ``-- foo`` is encoded as
    ``--- foo`` — same three-dash-space prefix. We disambiguate with
    two checks:

    1. The header must match ``--- a/<path>`` or ``--- /dev/null``.
       Arbitrary text after ``--- `` (``--- arbitrary deletion``) is
       not a file header shape for the patches SWE-bench emits.
    2. A file header is always immediately followed by ``+++ `` on
       the next line. A deletion of content that happens to match the
       ``--- a/`` shape would be followed by another body line (most
       commonly ``-<text>`` or `` <text>``), not ``+++ ``.
    """
    if not _FILE_HEADER_RE.match(line):
        return False
    if next_line is None:
        return False
    return next_line.startswith("+++ ")


def _split_file_diffs(patch: str) -> list[list[str]]:
    """Split a patch into per-file blocks.

    Boundary rules:

    * ``diff --git`` always starts a new block.
    * A ``--- `` line starts a new block when (a) it looks like a real
      file header (``--- a/<p>`` or ``--- /dev/null``), (b) the next
      line is ``+++ ``, and (c) either we are not yet in a block or
      the current block already emitted its own ``+++`` (so the
      previous file is complete).

    Rule (c) is what makes bare multi-file diffs work. Rule (b) is
    what prevents a hunk-body deletion like ``--- a/foo.py`` (the
    content ``-- a/foo.py`` being removed) from being misread as a
    new file header.
    """
    lines = patch.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    seen_plus_in_current = False

    def _starts_new_block(line: str, next_line: str | None) -> bool:
        if line.startswith("diff --git "):
            return True
        if _is_minus_file_header(line, next_line):
            if not current:
                return True
            if seen_plus_in_current:
                return True
        return False

    for i, line in enumerate(lines):
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if _starts_new_block(line, next_line):
            if current:
                blocks.append(current)
            current = [line]
            seen_plus_in_current = False
        else:
            current.append(line)
            if line.startswith("+++ "):
                seen_plus_in_current = True
    if current:
        blocks.append(current)
    return blocks


def _correct_file_diff(
    block: list[str], resolve_source,
) -> str | None:
    """Apply source/target-aware path correction to a single file block.

    Returns the rewritten block as a joined string, or ``None`` to
    reject the whole patch.
    """
    # First pass: determine the file-diff's shape. A "target is new"
    # block is one where the ``+++`` / rename-to / copy-to path does
    # not have to exist at base_commit and therefore must not be
    # oracle-checked.
    source_is_devnull = False
    is_rename = False
    is_copy = False
    for line in block:
        if line.startswith("--- /dev/null"):
            source_is_devnull = True
        elif line.startswith("rename from ") or line.startswith("rename to "):
            is_rename = True
        elif line.startswith("copy from ") or line.startswith("copy to "):
            is_copy = True

    target_is_new = source_is_devnull or is_rename or is_copy

    # Second pass: establish the source-side correction (if any) by
    # looking at the FIRST `--- a/<path>` before any `@@` hunk header.
    # (Lines matching `--- a/...` after a hunk header are hunk-body
    # deletions, not file headers.)
    source_rewrite: dict[str, str] = {}  # original path → corrected path
    for line in block:
        if line.startswith("@@"):
            break
        if line.startswith("--- a/"):
            path = line[len("--- a/"):]
            fixed = resolve_source(path)
            if fixed is None:
                return None
            if fixed != path:
                source_rewrite[path] = fixed
            break

    def _mirror(path: str) -> str:
        """Apply the same correction to a target path if we rewrote
        the source to a new file. Only used for plain modify (source
        == target)."""
        return source_rewrite.get(path, path)

    # Third pass: rewrite each line using the classification. File-
    # header transformations only apply BEFORE the first `@@` hunk
    # header. After that, every line is hunk-body content and must
    # pass through verbatim.
    in_hunk_body = False
    out: list[str] = []
    for line in block:
        if in_hunk_body:
            out.append(line)
            if line.startswith("@@"):
                # Next hunk header (same file) — still hunk-body phase
                # for the file-header rewriter's purposes.
                pass
            continue
        if line.startswith("@@"):
            in_hunk_body = True
            out.append(line)
            continue
        if line.startswith("diff --git "):
            m = re.match(r"^(diff --git )a/(\S+) b/(\S+)(.*)$", line)
            if not m:
                out.append(line)
                continue
            prefix, a_path, b_path, rest = m.groups()
            # `a/` side: always source-like; correct it. (For create
            # patches, the diff --git line still carries a placeholder
            # a/new.py b/new.py where "a/new.py" does not yet exist; in
            # that case we still accept it without correction — the
            # /dev/null marker on `--- /dev/null` is what actually
            # declares a create.)
            if source_is_devnull:
                # Both sides are placeholders for the new path; pass through.
                out.append(line)
                continue
            fa = resolve_source(a_path)
            if fa is None:
                return None
            # `b/` side: if target is new (rename/copy to), pass through.
            # Otherwise mirror the source correction.
            if is_rename or is_copy:
                fb = b_path
            else:
                fb = _mirror(b_path) if b_path == a_path else b_path
            out.append(f"{prefix}a/{fa} b/{fb}{rest}")
            continue
        if line.startswith("--- a/"):
            path = line[len("--- a/"):]
            fixed = resolve_source(path)
            if fixed is None:
                return None
            out.append("--- a/" + fixed)
            continue
        if line.startswith("--- ") or line.startswith("+++ /dev/null"):
            out.append(line)
            continue
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):]
            if target_is_new:
                out.append(line)  # new file / rename target / copy target
            else:
                # Plain modify: mirror the source correction.
                out.append("+++ b/" + _mirror(path))
            continue
        if line.startswith("rename from ") or line.startswith("copy from "):
            prefix = "rename from " if line.startswith("rename from ") else "copy from "
            path = line[len(prefix):]
            fixed = resolve_source(path)
            if fixed is None:
                return None
            out.append(prefix + fixed)
            continue
        if line.startswith("rename to ") or line.startswith("copy to "):
            # Target of rename/copy is new by definition — pass through.
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


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
            # A ``---`` line inside a hunk body is ambiguous: it could
            # be either a real file header (next file in a bare
            # multi-file diff) or a deletion of content that starts
            # with ``-- `` (encoded as ``--- ...`` on the wire). We
            # disambiguate with lookahead — a real file header is
            # always followed by ``+++ `` on the next line. See
            # ``_is_minus_file_header`` for the shared helper.
            body_old = 0
            body_new = 0
            i += 1
            while i < len(lines):
                body_line = lines[i]
                next_line = lines[i + 1] if i + 1 < len(lines) else None
                if body_line.startswith("@@"):
                    break
                if _is_minus_file_header(body_line, next_line):
                    break
                if body_line.startswith("+++ ") and not body_line.startswith("+++ b/") and body_line != "+++ /dev/null":
                    # Safety: stray +++ header without sibling --- is malformed.
                    break
                if body_line.startswith(_METADATA_PREFIXES):
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
