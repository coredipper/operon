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


#: Machine-readable rejection reason codes returned by
#: :func:`sanitize_with_reason`. Each code names a specific failure
#: mode the format-correction retry loop in ``eval.swebench_phase2``
#: can build a targeted re-ask from. Kept as a tuple rather than an
#: Enum so it's trivially JSON-serialisable into artifacts.
SANITIZE_REASONS: tuple[str, ...] = (
    "",                      # success
    "empty_extraction",      # no usable diff content
    "placeholder_hunk",      # @@ header with XXX / N / ? line numbers
    "truncated_hunk",        # body shorter than declared counts
    "overlong_hunk",         # body longer than declared counts, bad boundary
    "malformed_metadata",    # @@ header couldn't be parsed at all
    "path_not_found",        # source path missing + no basename match
    "ambiguous_path",        # source path missing + >1 basename matches
)


def sanitize(
    patch: str, repo_slug: str,
    *, tree_paths: frozenset[str] | None = None,
) -> str:
    """Return a sanitized unified diff, or ``""`` if unsalvageable.

    Thin wrapper over :func:`sanitize_with_reason` that discards the
    reason code. Kept so callers that only care about the cleaned
    patch (the majority of existing callers and tests) don't need to
    unpack a tuple. New callers that want to route the failure reason
    to a retry prompt should call :func:`sanitize_with_reason` directly.
    """
    patch_out, _reason = sanitize_with_reason(
        patch, repo_slug, tree_paths=tree_paths,
    )
    return patch_out


def sanitize_with_reason(
    patch: str, repo_slug: str,
    *, tree_paths: frozenset[str] | None = None,
) -> "tuple[str, str]":
    """Return ``(sanitized_patch, reason)``.

    On success, ``sanitized_patch`` is the cleaned, newline-terminated
    diff and ``reason`` is ``""``. On failure, ``sanitized_patch`` is
    ``""`` and ``reason`` is one of :data:`SANITIZE_REASONS` excluding
    the empty success code. The reason is the seam the format-
    correction retry loop uses to build a targeted re-ask: a
    ``"placeholder_hunk"`` can be fixed by asking for real line
    numbers; a ``"path_not_found"`` needs a different file; etc.

    ``repo_slug`` is the SWE-bench ``repo`` field, e.g. ``"django/django"``.
    ``tree_paths``, when provided, is the set of repo-relative file
    paths that actually exist at ``base_commit``. When ``None``, no
    oracle-based path correction runs.
    """
    if not patch or not patch.strip():
        return "", "empty_extraction"

    normalized = _normalize_paths(patch, repo_slug)
    normalized = _repair_bare_empty_context(normalized)
    if tree_paths is not None:
        normalized, reason = _fuzzy_correct_paths_with_reason(
            normalized, tree_paths,
        )
        if not normalized:
            return "", reason
    ok, reason = _validate_hunks_with_reason(normalized)
    if not ok:
        return "", reason

    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized, ""


def _fuzzy_correct_paths(patch: str, tree_paths: frozenset[str]) -> str:
    """Backward-compat wrapper returning just the corrected patch.

    New callers should prefer :func:`_fuzzy_correct_paths_with_reason`
    so they can distinguish ``path_not_found`` from ``ambiguous_path``
    and feed that reason into a retry prompt.
    """
    corrected, _reason = _fuzzy_correct_paths_with_reason(patch, tree_paths)
    return corrected


def _fuzzy_correct_paths_with_reason(
    patch: str, tree_paths: frozenset[str],
) -> "tuple[str, str]":
    """Like :func:`_fuzzy_correct_paths` but also returns a reason
    code when the correction fails.

    Returns ``(corrected_patch, "")`` on success. On failure, returns
    ``("", "path_not_found")`` if no basename match was found, or
    ``("", "ambiguous_path")`` if multiple basename matches made the
    rewrite ambiguous. Per-file-diff: the first failure reason wins —
    earlier blocks get priority.
    """
    basename_index: dict[str, list[str]] = {}
    for p in tree_paths:
        base = p.rsplit("/", 1)[-1]
        basename_index.setdefault(base, []).append(p)

    # Track the reason the most recent ``_resolve_source`` call came
    # back empty. Closed over by the nested function so we can surface
    # it up to the file-diff loop without changing the resolver's
    # signature (which is consumed by _correct_file_diff unchanged).
    resolve_failure_reason: list[str] = [""]

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
        if len(matches) == 0:
            resolve_failure_reason[0] = "path_not_found"
        else:
            resolve_failure_reason[0] = "ambiguous_path"
        return None

    blocks = _split_file_diffs(patch)
    out_parts: list[str] = []
    for block in blocks:
        fixed = _correct_file_diff(block, _resolve_source)
        if fixed is None:
            # resolve_failure_reason is set by the resolver; guard
            # against an unexpected None from elsewhere in the block
            # by defaulting to path_not_found (the most common case).
            return "", (resolve_failure_reason[0] or "path_not_found")
        out_parts.append(fixed)
    return "\n".join(out_parts), ""


_MINUS_FILE_HEADER_RE = re.compile(r"^--- (?:a/|/dev/null(?:$|\t))")
_PLUS_FILE_HEADER_RE = re.compile(r"^\+\+\+ (?:b/|/dev/null(?:$|\t))")


def _is_plus_file_header(line: str) -> bool:
    """Return True iff ``line`` looks like a real ``+++`` file header.

    Shape-only check: ``+++ b/<path>`` or ``+++ /dev/null`` (optionally
    followed by a tab + timestamp). Shape alone is NOT sufficient to
    decide if a line inside a hunk body is a header — content shaped
    ``++ b/...`` serialises on the wire as ``+++ b/...`` and collides
    with this pattern. Callers must use hunk-count tracking (see
    ``_scan_hunk_extent``) to know whether a line is in a body or
    at a file boundary.
    """
    return bool(_PLUS_FILE_HEADER_RE.match(line))


def _is_minus_file_header(line: str, next_line: str | None) -> bool:
    """Shape-only check that ``line`` looks like a ``---`` file header
    paired with a ``+++`` header on the next line.

    Same caveat as :func:`_is_plus_file_header`: position matters.
    Inside a hunk body, a deletion of content shaped ``-- a/...``
    followed by an addition shaped ``++ b/...`` will satisfy this
    shape check too. Callers must use hunk-count tracking to know
    they are outside a hunk body.
    """
    if not _MINUS_FILE_HEADER_RE.match(line):
        return False
    if next_line is None:
        return False
    return _is_plus_file_header(next_line)


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def _parse_hunk_counts(line: str) -> "tuple[int, int] | None":
    """Return ``(old_count, new_count)`` for a well-formed hunk header.

    Returns ``None`` if the header has placeholder line numbers (``XXX``,
    ``N``, etc.) or otherwise fails to parse. The count defaults to 1
    when omitted (``@@ -42 +42 @@`` is valid git syntax).
    """
    m = _HUNK_HEADER_RE.match(line)
    if not m:
        return None
    old = int(m.group("old_count")) if m.group("old_count") else 1
    new = int(m.group("new_count")) if m.group("new_count") else 1
    return old, new


def _scan_hunk_extent(
    lines: list[str], start: int, old_count: int, new_count: int,
) -> "tuple[int, bool]":
    """Walk ``lines[start:]`` consuming the hunk body until counts drain.

    Returns ``(end_idx, ok)``. ``end_idx`` is the index of the first
    line *after* the hunk body (or ``len(lines)`` if truncated). ``ok``
    is True iff:

    1. The body was well-formed and both counts reached zero.
    2. The line at ``end_idx`` is a legal post-hunk boundary —
       another ``@@`` hunk header, a ``---``/``+++`` file header,
       ``diff --git``, a git metadata line, or EOF. This second
       check catches **overlong** hunks: a declared ``@@ -1 +1 @@``
       whose body contains ``-old``, ``+new``, ``+extra`` must be
       rejected, because git apply would choke on the dangling
       ``+extra``. Review #735.

    Inside a hunk body, a line's role is determined by its first
    character:
      * ``+`` → addition (decrements ``new_count``)
      * ``-`` → deletion (decrements ``old_count``)
      * `` `` or empty → context (decrements both; empty is a
        whitespace-stripped blank that the repair pass will fix up)
      * ``\\ No newline at end of file`` → marker, does not count.
        Accepted both inside the body (as a marker for the preceding
        ``+``/``-`` line) and as the first line after the body
        (as a marker for the hunk's final body line).
      * anything else → malformed; stop with ``ok=False``.

    An in-body line whose content is shaped like a file header
    (``--- a/...``, ``+++ b/...``) is STILL interpreted by its first
    character — a ``-``-prefixed line is always a delete inside a
    hunk body regardless of what follows.
    """
    i = start
    while i < len(lines) and (old_count > 0 or new_count > 0):
        line = lines[i]
        if line == r"\ No newline at end of file":
            i += 1
            continue
        if line.startswith("+"):
            new_count -= 1
        elif line.startswith("-"):
            old_count -= 1
        elif line.startswith(" ") or line == "":
            old_count -= 1
            new_count -= 1
        else:
            return i, False
        i += 1
    if old_count != 0 or new_count != 0:
        return i, False

    # Counts drained. Consume a trailing ``\ No newline at end of file``
    # marker if present (it applies to the hunk's final body line).
    if i < len(lines) and lines[i] == r"\ No newline at end of file":
        i += 1

    # The next line (if any) must be a legal post-hunk boundary, not
    # more body-shaped content. Body-shaped content after counts drain
    # means the hunk is overlong.
    if _is_post_hunk_boundary(lines, i):
        return i, True
    return i, False


def _is_post_hunk_boundary(lines: list[str], i: int) -> bool:
    """Return True iff ``lines[i]`` is a legal post-hunk boundary.

    Review #736: shape-only acceptance of ``---``/``+++`` lets overlong
    body content that happens to match header shape slip through. The
    correct disambiguation is structural: the current hunk has ended,
    and we're looking for where the NEXT hunk or file begins.

    Legal boundaries (in decreasing order of specificity):

    1. EOF (``i >= len(lines)``).
    2. ``@@ ...`` — another hunk of the same file. Unambiguous: no
       body line starts with ``@@``.
    3. ``diff --git ...`` — a new file-diff. Unambiguous: no body line
       starts with this exact prefix.
    4. A git metadata line (``index ``, ``new file mode``, etc.) —
       these sit BETWEEN ``diff --git`` and the ``---``/``+++`` pair
       of a new file-diff, so finding one here implies a new file.
    5. The three-line triplet ``--- a/<p>`` + ``+++ b/<p>`` + ``@@`` —
       a bare multi-file diff's next file-diff starts. Requiring the
       ``@@`` on line ``i+2`` is what rules out body-content
       impostors: a body pair ``-- a/...`` / ``++ b/...`` followed by
       arbitrary body content will not put ``@@`` at position ``i+2``.

    Any other shape (including shape-only ``---``/``+++`` without the
    full triplet) is treated as overlong body content, which rejects
    the whole patch.
    """
    if i >= len(lines):
        return True
    line = lines[i]
    if line.startswith("@@"):
        return True
    if line.startswith("diff --git "):
        return True
    if line.startswith(_METADATA_PREFIXES):
        return True
    # Bare multi-file triplet: ---, +++, @@.
    if _MINUS_FILE_HEADER_RE.match(line):
        plus = lines[i + 1] if i + 1 < len(lines) else None
        at = lines[i + 2] if i + 2 < len(lines) else None
        if plus is not None and _PLUS_FILE_HEADER_RE.match(plus):
            if at is not None and at.startswith("@@"):
                return True
    return False


def _split_file_diffs(patch: str) -> list[list[str]]:
    """Split a patch into per-file blocks.

    The key invariant is that lines *inside* a hunk body are never
    interpreted as file headers, no matter what they look like. We
    track hunk state via the declared ``@@ -a,b +c,d @@`` counts and
    walk the body using :func:`_scan_hunk_extent`. Outside hunks,
    ``diff --git`` is an unambiguous boundary and ``--- a/<p>`` paired
    with ``+++ b/<p>`` on the next line is a file-header pair; we
    split on it when we're between files (the current block either is
    empty or has already emitted a ``+++`` header, meaning the
    previous file's definition is complete).
    """
    lines = patch.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    seen_plus_in_current = False

    i = 0
    while i < len(lines):
        line = lines[i]
        # Inside a hunk body? Consume it wholesale — no boundary check.
        if line.startswith("@@"):
            counts = _parse_hunk_counts(line)
            if counts is not None:
                old, new = counts
                current.append(line)
                end, _ok = _scan_hunk_extent(lines, i + 1, old, new)
                for k in range(i + 1, end):
                    current.append(lines[k])
                i = end
                continue
            # Malformed hunk header — just append and move on. The
            # validator will reject downstream.
            current.append(line)
            i += 1
            continue

        # Outside hunk: look for boundaries.
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = [line]
            seen_plus_in_current = False
            i += 1
            continue
        if _is_minus_file_header(line, next_line):
            if not current or seen_plus_in_current:
                if current:
                    blocks.append(current)
                current = [line]
                seen_plus_in_current = False
                i += 1
                continue
        if line.startswith("+++ "):
            seen_plus_in_current = True
        current.append(line)
        i += 1
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

    Some editors/tools strip trailing whitespace, turning a valid
    context line (a single space) into a bare empty line. ``git apply``
    rejects bare empty lines inside hunks; we rewrite them back to
    ``" "``.

    Hunk boundaries come from the declared ``@@ -a,b +c,d @@`` counts
    (via :func:`_scan_hunk_extent`), not from line-shape heuristics.
    Review #733: shape-based heuristics can false-match adversarial
    hunk content shaped like ``--- a/...`` or ``+++ b/...``. Count-
    driven consumption is robust against such content.
    """
    lines = patch.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            counts = _parse_hunk_counts(line)
            if counts is None:
                out.append(line)
                i += 1
                continue
            old, new = counts
            out.append(line)
            # Walk body lines, repairing blanks; stop at exact count.
            end, _ok = _scan_hunk_extent(lines, i + 1, old, new)
            for k in range(i + 1, end):
                body_line = lines[k]
                if body_line == "":
                    out.append(" ")
                else:
                    out.append(body_line)
            i = end
            continue
        out.append(line)
        i += 1
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
    """Backward-compat wrapper returning just the ok/not-ok flag.

    New callers should prefer :func:`_validate_hunks_with_reason`
    so they can distinguish ``placeholder_hunk`` from ``truncated_hunk``
    from ``overlong_hunk`` and feed that reason into a retry prompt.
    """
    ok, _reason = _validate_hunks_with_reason(patch)
    return ok


def _validate_hunks_with_reason(patch: str) -> "tuple[bool, str]":
    """Like :func:`_validate_hunks` but also returns a reason code on
    failure.

    Returns ``(True, "")`` on success. On failure, returns
    ``(False, reason)`` where reason is one of ``placeholder_hunk``,
    ``truncated_hunk``, ``overlong_hunk``, ``malformed_metadata``, or
    ``empty_extraction`` (the last for a diff with no ``@@`` hunks at
    all — from the retry loop's perspective, that's equivalent to an
    empty extraction since the model didn't actually produce a diff).
    """
    lines = patch.splitlines()
    i = 0
    saw_any_hunk = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            saw_any_hunk = True
            m_raw = _HUNK_RE.match(line)
            if not m_raw:
                return False, "malformed_metadata"
            for key in ("old_start", "old_count", "new_start", "new_count"):
                val = m_raw.group(key)
                if val is None:
                    continue
                if not val.isdigit():
                    return False, "placeholder_hunk"
            counts = _parse_hunk_counts(line)
            if counts is None:
                return False, "placeholder_hunk"
            old, new = counts
            end, ok = _scan_hunk_extent(lines, i + 1, old, new)
            if not ok:
                # Distinguish truncated (body drained short) from
                # overlong (counts drained but next line isn't a
                # legal post-hunk boundary). _scan_hunk_extent returns
                # False in both cases; we re-check the counts-vs-end
                # shape here to pick the right reason for the retry
                # prompt.
                if end >= len(lines):
                    # Ran off the end without draining counts — pure
                    # truncation.
                    return False, "truncated_hunk"
                # We have a non-boundary line still inside what should
                # be post-hunk space, OR the walk hit malformed content
                # mid-body. Distinguish by probing one more time: if
                # re-scanning from the start with the declared counts
                # would have drained at ``end`` exactly, this is an
                # overlong case; otherwise a truncated / malformed body.
                if _is_post_hunk_boundary(lines, end) is False and (
                    lines[end].startswith(("+", "-", " "))
                ):
                    # Body-shaped content after counts should have drained:
                    # overlong.
                    return False, "overlong_hunk"
                return False, "truncated_hunk"
            i = end
            continue
        i += 1

    if not saw_any_hunk:
        return False, "empty_extraction"
    return True, ""
