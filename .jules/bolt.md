## 2025-02-28 - Compile Regexes in Class Definitions for Performance
**Learning:** Frequent instantiations of classes (e.g. `mRNA` in `ribosome.py`) with methods that repeatedly call inline `re.finditer` with string patterns are a significant performance bottleneck.
**Action:** Always pre-compile these regular expressions as class-level variables (using `typing.ClassVar` on dataclasses to avoid interfering with initialization) and refer to the compiled patterns. This yielded a ~42% performance boost in this instance.
