## 2024-05-24 - Pre-compiling Regex in Template Engines
**Learning:** In the `operon_ai` project, template parsing and variable interpolation in the `Ribosome` organelle used string-based `re.sub` and `re.finditer` on every call. Explicitly pre-compiling regexes as `typing.ClassVar` attributes avoids dictionary cache lookup overhead in hot paths, leading to a measurable ~16% speedup in template rendering benchmarks.
**Action:** Always check template parsing loops or hot paths involving regular expressions. Pre-compile them using `ClassVar` on the class level for maximum efficiency.

## 2024-05-23 - Pre-compile Regexes in dataclasses using ClassVar
**Learning:** When optimizing repetitive Regex compilation inside `dataclass` methods, simply assigning `re.compile()` to a class variable will treat it as a dataclass field by default. This alters the class constructor and creates errors.
**Action:** Use `typing.ClassVar` to correctly type-hint pre-compiled regular expression patterns as class attributes when using `@dataclass`, ensuring they are omitted from the generated `__init__` constructor.
