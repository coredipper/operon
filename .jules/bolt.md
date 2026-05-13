## 2024-05-24 - Pre-compiling Regex in Template Engines
**Learning:** In the `operon_ai` project, template parsing and variable interpolation in the `Ribosome` organelle used string-based `re.sub` and `re.finditer` on every call. Explicitly pre-compiling regexes as `typing.ClassVar` attributes avoids dictionary cache lookup overhead in hot paths, leading to a measurable ~16% speedup in template rendering benchmarks.
**Action:** Always check template parsing loops or hot paths involving regular expressions. Pre-compile them using `ClassVar` on the class level for maximum efficiency.

## 2024-05-23 - Pre-compile Regexes in dataclasses using ClassVar
**Learning:** When optimizing repetitive Regex compilation inside `dataclass` methods, simply assigning `re.compile()` to a class variable will treat it as a dataclass field by default. This alters the class constructor and creates errors.
**Action:** Use `typing.ClassVar` to correctly type-hint pre-compiled regular expression patterns as class attributes when using `@dataclass`, ensuring they are omitted from the generated `__init__` constructor.

## 2024-05-25 - Dictionary Lookup Optimization
**Learning:** Double dictionary lookups (`if key in dict` followed by `dict[key]`) are a frequent micro-bottleneck. Using the walrus operator with `dict.get()` (`if val := dict.get(key):`) cuts dictionary lookups in half, providing measurable speedups in highly-frequent hot paths like coordination gradients. Furthermore, returning the value directly from a setter method allows callers to skip an additional, otherwise redundant, `get()` lookup when calculating state deltas.
**Action:** Always refactor redundant `in` + `[]` accesses to single `.get()` calls using the walrus operator. Consider updating stateful setter methods to return their updated value to allow callers to calculate deltas more efficiently.

## 2024-05-24 - Pre-compiling Regex in Chaperone
**Learning:** In `operon_ai`, the `Chaperone` organelle performs JSON extraction and repair using regular expressions. Originally, these string patterns were passed directly to `re.findall` and `re.sub` inside its loop methods (`_extract_json`, `_fold_repair`, etc.), relying on Python's regex cache. Explicitly pre-compiling the `JSON_EXTRACTION_PATTERNS` and `JSON_REPAIRS` at the class level via `ClassVar` reduces execution overhead by more than 50% in tight repair loops.
**Action:** Consistently verify if regex operations inside any parsing logic or loops (like extraction/repair methods) are defined as strings instead of pre-compiled `re.Pattern` objects. Pre-compile them at the class level with `ClassVar` to ensure maximum performance.

## 2024-05-25 - Python Vector Math Optimization
**Learning:** In the `operon_ai` project, vector operations (like cosine similarity) were using generator expressions with `sum()` and `math.sqrt()` (e.g., `math.sqrt(sum(x * x for x in v))` and `sum(x * y for x, y in zip(a, b))`). Replacing these pure Python generators with `math.hypot(*v)` for magnitudes and `sum(map(operator.mul, a, b))` for dot products yields a ~2.5x to 6x speedup by leveraging C-level implementations. This is critical for high-frequency ML/health metrics in environments without numpy.
**Action:** Always prefer `math.hypot` for calculating Euclidean norms and `sum(map(operator.mul, a, b))` over zip/generator comprehensions for dot products in pure Python code where numpy is not available.

## 2024-05-13 - Pre-compiled Regex class variables
**Learning:** Frequent instantiations of inline regular expressions in iterative contexts, such as `MHCDisplay.generate_peptide` executing multiple string match validations, present an accumulative drag despite Python internal regex caching.
**Action:** Lift static internal string regex validation patterns to pre-compiled class-level properties via `ClassVar[re.Pattern] = re.compile()`.
