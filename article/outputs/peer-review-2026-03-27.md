# Peer Review: "Biological Motifs for Agentic Control"

**Date:** 2026-03-27
**Reviewer:** Feynman peer-review skill (automated, 3-agent panel)
**Scope:** Full paper — citations, proofs, structure, writing

---

## Verdict

The paper presents a typed interface correspondence between gene regulatory
networks and agentic software architectures, formalized via polynomial
functors and wiring diagrams. The mathematical content is sound — all proofs
are correct in their core arguments. The main weaknesses are structural
(Section 6 is overloaded, theorem formatting was inconsistent) and
presentational (empirical calibration overstates validation strength,
rhetorical passages occasionally overreach).

**Recommendation:** Major revision (structural reorganization of Section 6,
empirical framing clarification).

---

## Issues Fixed in This Pass

### HIGH — Theorem/Definition Environment Formatting
- **04-operad.tex**: Theorems 1 and 2 used `\paragraph{Theorem N (...)}` instead of `\begin{theorem}`, causing `\ref` to resolve to the parent section number instead of the theorem number. Both now use proper `\begin{theorem}[...]...\end{theorem}` environments with labels.
- **04-operad.tex**: Corollary used `\paragraph{}` with no label. Now uses `\begin{corollary}[Correlation Degradation]\label{cor:correlation-degradation}`.
- **04-operad.tex**: Both proof sketches used `\paragraph{Proof sketch.}`. Now use `\begin{proof}[Proof sketch]...\end{proof}`.
- **03-mapping.tex**: Definitions 1 and 2 used `\paragraph{Definition N (...)}`. Now use `\begin{definition}[...]\label{def:...}...\end{definition}`.

### HIGH — Wrong Section Reference
- **02-related-work.tex:49** and **06-discussion.tex:321**: §8.4 referred to bi-temporal memory, but §8.4 is "Trust and Provenance". Fixed to §8.8.

### MEDIUM — Undefined Citations
- **09-conclusion.tex:133**: `\cite{animaworks}` and `\cite{swarms}` had no BibTeX entries. Added both to `references.bib`.

### MEDIUM — Year Mismatch in BibTeX
- **references.bib**: `abbott2023neural` has arXiv ID 2402 (Feb 2024). Renamed to `abbott2024neural` with `year={2024}`. Updated cite key in `07-optimization.tex`.

### MEDIUM — Proof Claim Too Strong
- **06-discussion.tex:429**: Parallel Acceleration theorem part (c) claimed quality loss "proportional to" mutual information. Weakened to "bounded below by a function of" — the proof sketch demonstrates qualitative loss but not the proportionality constant.

### MEDIUM — Planning Cost Derivation Unclear
- **06-discussion.tex:503**: $O(t \cdot n)$ planning cost explanation was hand-wavy about the transition from model size to query count. Clarified: the planner evaluates all $n$ agents' capability profiles for each of $t$ subgoals.

### MEDIUM — Terminology Inconsistency
- **06-discussion.tex:567**: "This refines our Isomorphism" contradicts the paper's own disclaimer that the mapping is a correspondence, not an isomorphism. Fixed to "correspondence".

### LOW — Hardcoded Table Reference
- **03-mapping.tex:161**: "Table~1" was hardcoded. Added `\label{tab:correspondence-dictionary}` and converted to `Table~\ref{...}`.

### LOW — Hardcoded Definition References
- **03-mapping.tex:67**: "Definitions~1 and~2" was hardcoded. Converted to `Definitions~\ref{def:gene-object} and~\ref{def:agent-capability}`.

### LOW — Missing BibTeX Verifiability
- **references.bib**: `davis2026reasoning` (UCLA seminar) had no URL or howpublished. Added `howpublished={UCLA CS 201-A Seminar}`.

### LOW — Redundant Correlation Name
- **04-operad.tex:94**: "Pearson/phi" was redundant for Bernoulli variables. Simplified to "phi coefficient".

---

## Issues Noted but Not Fixed (Require Authorial Decision)

### HIGH — Section 6 Structural Overload
Section 6 ("Discussion") contains ~650 lines and includes the majority of
the paper's formal contributions: four epistemic topology theorems with
proofs, morphogen diffusion formalization, multi-cellular organization,
adaptive assembly, cognitive modes, sleep consolidation, social learning,
and developmental staging. Core technical contributions should not be buried
in a Discussion section. **Recommendation:** Promote epistemic topology to
its own section; move multi-cellular organization into the formal framework
sections.

### HIGH — Empirical Calibration Framing
The calibration against Kim et al. (2025) is qualitative — directional
consistency, not quantitative prediction. The theorems formalize known design
intuitions (more agents = more errors, sequential = handoff cost). Table 3
juxtaposing "Prediction" and "Observed" columns implies stronger validation
than the evidence supports. **Recommendation:** Frame as "formalizing known
design intuitions" rather than "validated predictions."

### MEDIUM — Abstract Density
The abstract packs ~13 contributions, six layers, and implementation metrics
into one paragraph. **Recommendation:** Split into problem/approach/results;
lead with 2-3 core contributions.

### MEDIUM — 13 Listed Contributions
Most venues expect 3-5 contributions per paper. **Recommendation:** Group
into 3-4 high-level themes.

### MEDIUM — Editorial Content in Related Work
Section 2 line 49 contains original editorial claims about Operon's
bi-temporal approach. This belongs in the methods sections.

### MEDIUM — Synthetic Evaluation Framing
Wilson 95% CIs over deterministic synthetic corruptions measure
implementation correctness, not real-world reliability. The 100% immune
sensitivity is expected when attacks are designed to be detectable.
**Recommendation:** Frame as unit-test verification, not performance
benchmarks.

### MEDIUM — Rhetorical Overreach
"The Operon framework is not merely a safety feature; it is the necessary
evolutionary adaptation" (06-discussion.tex:617) is promotional, not
analytical. The Cambrian Parallel (lines 611-615) is speculative.
**Recommendation:** Use conditional language ("we hypothesize...").

### MEDIUM — Repeated Content
Metabolic Bound appears in three locations (03-mapping, 06-discussion,
08-implementation). Bi-temporal memory in four locations. Some redundancy
aids readability; the Section 6 recapitulation adds little.

### LOW — Version Numbers in Conclusion
09-conclusion.tex references v0.17-v0.23. Unusual for a research paper.

### LOW — Dangling Example References
"Example 69", "Example 71", etc. reference repository scripts, not paper
content.

### LOW — Sparse Figures
3 figures for a paper claiming 13 contributions across category theory,
biology, and multi-agent systems. No architecture overview diagram.

### LOW — Missing Section Transitions
Sections 5→6 and 6→7 lack connecting paragraphs.

---

## Bibliography Health

- **0 undefined citations** (after fixes)
- **0 unused bib entries**
- **3 year-mismatch cite keys**: `picard2022mips` (actual 2018),
  `chandel2024mitochondria` (actual 2014), `abbott2024neural` (fixed from
  2023→2024)
- **~8 arXiv entries** missing `eprint`/`archivePrefix` fields (inconsistent
  with `abbott2025flashattention` which has them)

## Mathematical Soundness

All proofs are **correct** in their core arguments:
- Theorem 1 (CFFL Error Suppression): Correct. Fréchet-Hoeffding bounds properly applied.
- Theorem 2 (Injection Resistance): Correct. Deliberately narrow scope is well-acknowledged.
- Error Amplification Bound: Correct. Standard union bound.
- Sequential Coordination Penalty: Correct. DPI properly applied.
- Parallel Acceleration: Correct with one weakened claim (proportional→bounded).
- Tool Density Scaling: Correct with clarified derivation.

---

## Paper Builds Cleanly

`tectonic main.tex` — no undefined references, no undefined citations.
Only cosmetic overfull hbox warnings remain.
