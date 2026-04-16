"""Tests for eval/_patch_extraction.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval._patch_extraction import extract_patch


def test_fenced_diff_block():
    output = """Here is the fix:

```diff
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
-print("old")
+print("new")
 x = 1
```
"""
    patch = extract_patch(output)
    assert patch.startswith("--- a/foo.py")
    assert "+print(\"new\")" in patch


def test_fenced_patch_block():
    output = """```patch
--- a/bar.py
+++ b/bar.py
@@ -10,7 +10,7 @@
-    return None
+    return 0
```"""
    patch = extract_patch(output)
    assert "bar.py" in patch
    assert "+    return 0" in patch


def test_bare_unified_diff():
    output = """I analyzed the code. Here's the fix:

--- a/baz.py
+++ b/baz.py
@@ -5,3 +5,3 @@
 def f():
-    pass
+    return 1

Done."""
    patch = extract_patch(output)
    assert patch.startswith("--- a/baz.py")
    assert "+    return 1" in patch


def test_git_diff_header():
    output = """```diff
diff --git a/qux.py b/qux.py
index abc..def 100644
--- a/qux.py
+++ b/qux.py
@@ -1 +1,2 @@
 pass
+import sys
```"""
    patch = extract_patch(output)
    assert "diff --git" in patch or "qux.py" in patch


def test_stage_formatted_output():
    output = """[localize]
The bug is in foo.py line 42.

[edit]
```diff
--- a/foo.py
+++ b/foo.py
@@ -40,5 +40,5 @@
-    x = None
+    x = 0
```

[verify]
Looks good."""
    patch = extract_patch(output)
    assert "+    x = 0" in patch
    # Should NOT include the [verify] prose
    assert "Looks good" not in patch


def test_empty_output():
    assert extract_patch("") == ""


def test_prose_only():
    assert extract_patch("I cannot fix this bug.") == ""


def test_no_diff_just_code():
    # Code blocks without diff format should return empty
    output = """```python
def foo():
    return 1
```"""
    assert extract_patch(output) == ""


def test_multi_file_diff():
    output = """--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-foo
+bar"""
    patch = extract_patch(output)
    assert "a.py" in patch
    assert "b.py" in patch
