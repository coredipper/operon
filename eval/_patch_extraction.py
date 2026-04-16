"""Extract unified diffs from LLM / organism outputs.

Handles:
- ```diff ... ``` fenced blocks
- ```patch ... ``` fenced blocks
- Bare unified diffs (--- a/file, +++ b/file)
- Multi-stage outputs like [edit]\\n<diff>\\n[verify]\\n...
- Refusals / empty outputs (returns "")
"""

from __future__ import annotations

import re


# Matches [edit] or [patch] stage section from organism output
_STAGE_EDIT = re.compile(r"\[(?:edit|patch)\]\n(.*?)(?=\n\[|\Z)", re.DOTALL)


def extract_patch(output: str) -> str:
    """Return the best-effort unified diff from *output*.

    Returns empty string if no diff is found (organism refused,
    output contains only prose, etc.).
    """
    if not output or not output.strip():
        return ""

    # Strategy 1: look inside [edit] / [patch] stage first
    stage_match = _STAGE_EDIT.search(output)
    candidates = [stage_match.group(1)] if stage_match else []
    candidates.append(output)

    for text in candidates:
        # Fenced diff blocks
        fenced = _extract_fenced(text)
        if fenced:
            return fenced
        # Bare unified diff
        bare = _extract_bare(text)
        if bare:
            return bare

    return ""


def _extract_fenced(text: str) -> str:
    # Find all fenced blocks, pick the first that looks like a diff
    for m in re.finditer(r"```(?:diff|patch)?\n(.*?)\n```", text, re.DOTALL):
        body = m.group(1)
        if _looks_like_diff(body):
            return body.strip() + "\n"
    return ""


_GIT_METADATA_PREFIXES = (
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


def _extract_bare(text: str) -> str:
    # Concatenate all diff-looking chunks
    chunks = []
    current: list[str] = []
    in_diff = False
    for line in text.splitlines():
        if (line.startswith("diff --git ")
                or line.startswith("--- a/")
                or line.startswith("--- /dev/null")):
            if current and in_diff:
                chunks.append("\n".join(current))
            current = [line]
            in_diff = True
        elif in_diff:
            if (line.startswith("+++") or line.startswith("@@") or
                    line.startswith("+") or line.startswith("-") or
                    line.startswith(" ") or
                    line == "\\ No newline at end of file" or
                    line.startswith(_GIT_METADATA_PREFIXES)):
                current.append(line)
            else:
                if current:
                    chunks.append("\n".join(current))
                current = []
                in_diff = False
    if current and in_diff:
        chunks.append("\n".join(current))

    result = "\n".join(chunks).strip()
    return result + "\n" if result else ""


def _looks_like_diff(text: str) -> bool:
    """A block is a valid diff only if it has file headers, not just hunks.

    Hunk-only snippets (just ``@@ -1 +1 @@`` lines) cannot be applied by
    ``git apply``, so we reject them. Accepts ``---``/``+++`` pairs
    including ``/dev/null`` headers used for file add/delete, and also
    accepts ``diff --git`` as a standalone indicator.
    """
    has_minus_header = "--- a/" in text or "--- /dev/null" in text
    has_plus_header = "+++ b/" in text or "+++ /dev/null" in text
    return (has_minus_header and has_plus_header) or "diff --git " in text
