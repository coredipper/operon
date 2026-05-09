## 2024-05-24 - Pre-compiling Regular Expressions
**Learning:** For performance efficiency, frequently used regular expressions (such as those in the template engine or extraction logic) should be pre-compiled as class-level attributes using `typing.ClassVar`. This avoids redundant compilation overhead, especially within loops or highly recurrent methods.
**Action:** When defining patterns for string matching or data extraction, use `re.compile(pattern, flags)` directly in the class definition (or precompile them during initialization) instead of compiling them on the fly.
