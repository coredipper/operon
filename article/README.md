# Operon papers

Each `paperN/` directory is a standalone paper with its own `main.tex`,
`sections/`, and compiled `main.pdf`. The monograph (`monograph.tex` →
`sections/01-12`) is a separate long-form treatise, not a collation.

## Papers

| Dir | Title | Topic | Status |
|---|---|---|---|
| [`paper1`](paper1/) | Epistemic Topology for AI Agent Systems | Typed wiring diagrams + epistemic interpretation of coordination patterns; a design calculus for topology under reliability/cost constraints | Complete (12pp) |
| [`paper2`](paper2/) | Convergence by Composition | `ExternalTopology` adapter IR for 6 frameworks; 4 epistemic bounds, 4 TLA+ specs, co-design fixed-point proof | Complete (13pp) |
| [`paper3`](paper3/) | Do Biological Abstractions Generalize to Meta-Level AI Systems Engineering? | Evolution loop over Genome configs; negative result — bio abstractions generalize as code structure, not as optimization algorithms | Complete (5pp, short-form) |
| [`paper4`](paper4/) | Do Biological Structural Guarantees Earn Their Complexity? | Three deep benchmarks (metabolic gating, quorum sensing, Bayesian stagnation) vs. naive baselines at 10M+ trials | Complete (16pp, arXiv zip prepared) |
| [`paper5`](paper5/) | Harness Engineering as Categorical Architecture | ArchAgents triple $(G, \mathrm{Know}, \Phi)$ as formalization for agent harnesses; 100% certificate preservation across 5 compiler targets | Complete (15pp, arXiv zip committed in v0.34.4) |
| [`paper6`](paper6/) | Counterexample-Guided Reflective Evolution | GEPA with binary-plus-obligations evaluator vs scalar reward; Gemma 4, 10 seeds × 3 arms | WIP (10pp) — unresolved `\placeholder` in abstract (1) + results (10) |

## Monograph

- **`monograph.tex`** — modular entry point (`\input{sections/...}`), currently the active build on branch `docs/monograph-and-pdf-rebuild`.
- **`main.tex`** — older inline version of the same treatise; both title as *Biological Motifs for Agentic Control*.
- **`sections/01-introduction.tex` … `12-conclusion.tex`** — shared section files pulled in by `monograph.tex`.
- **`references.bib`** — shared bibliography for the monograph.

## Build

Each paper builds independently via `tectonic`:

```
cd article/paper5 && tectonic main.tex
```

The monograph builds from the repo's `article/` root:

```
cd article && tectonic monograph.tex
```

## arXiv artifacts

- `paper4/arxiv-submission.zip` — prepared, not yet in git (LaTeX build artifacts ignored per `.gitignore`)
- `paper5/arxiv-submission.zip` — committed in `v0.34.4` (`84c3bcd`)

Submission build commands for paper4/paper5 are captured in `.claude/settings.local.json` auto-allow rules.

## Why are the directories named `paperN`?

Historical + the names are load-bearing as stable public URLs. Published Medium and LinkedIn posts link to `https://github.com/coredipper/operon/blob/main/article/paper1/main.pdf`, so renaming the directories would 404 those external links. This index is the lookup table.
