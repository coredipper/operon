# External Agent Framework Integration — DSPy, GEPA, A2A

**Status:** design memo, paper-adjacent
**Version:** operon_ai 0.36.3
**Audience:** reviewers, contributors, downstream users
**Companion code:** `operon_ai/convergence/gepa_adapter.py`, `operon_ai/convergence/a2a_certificate.py`

---

## Abstract

Operon already converges with six agent-orchestration frameworks (Swarms, DeerFlow, AnimaWorks, Ralph, A-Evolve, Scion) via the bidirectional functor pair `parse_X_topology()` / `organism_to_X()` in `operon_ai/convergence/`. This memo extends the convergence story from *topologies* to *guarantees* by formalizing how Operon certificates travel across three additional surfaces: DSPy compiled artifacts (`L2`), GEPA reflective optimizers (`L2`), and A2A cross-vendor protocol messages (`L3`). Four theorems are stated. Two prototypes ship with this memo — a GEPA adapter and an A2A codec — and are directly callable from downstream code today.

---

## 1. Framework taxonomy

Every external agent framework consumes or produces a subset of three artifact classes:

| Layer | Artifact | Operon surface | Existing ($\checkmark$) / new |
|---|---|---|---|
| **L1 Runtime** | Topology / DAG of agent calls | `operon_ai/convergence/*_adapter.py` + `*_compiler.py` | $\checkmark$ (6 frameworks) |
| **L2 Artifact** | Compiled program or frozen prompt | `Certificate` attached at compile-time | **new** (DSPy, GEPA) |
| **L3 Protocol** | Wire message between opaque agents | `Certificate` as canonical `Part` | **new** (A2A) |

OpenMythos (`kyegomez/OpenMythos`) sits *below* L1 — it is a model architecture (Recurrent-Depth Transformer) — and is therefore out of scope for the agent-integration story; it could only appear as an `LLMProvider` backend, identical to any other model wrapper. Swarms is already handled at L1 and is included here only as the reference shape for the new adapters.

**L1 synthesis vs L1 mapping.** Conductor ([Nielsen et al. 2025, ICLR 2026, arXiv:2512.04388](https://arxiv.org/abs/2512.04388); Sakana AI) sits adjacent to the L1 row by *synthesising* topologies rather than mapping fixed ones — an RL-trained policy that designs both the communication graph and per-agent prompts, with recursive topology support enabling dynamic test-time scaling. Operon's L1 adapters (`parse_X_topology()` / `organism_to_X()`) treat topology as given; Conductor treats it as the learned object. See §2 Theorem 3 for the parallel "optimize-the-agent-layer" framing on the prompt axis (GEPA).

**Categorical formalization.** The L1/L2/L3 taxonomy above and the four certificate-transport theorems in §2 are the engineering instances of a categorical Architecture triple $(G, \mathrm{Know}, \Phi)$ developed in [Banu, *Harness Engineering as Categorical Architecture*, arXiv:2605.12239](https://arxiv.org/abs/2605.12239) (bib key `banu2026harness`). The four pillars of agent externalization (Memory, Skills, Protocols, Harness) map onto the triple's components: Memory as coalgebraic state, Skills as operad-composed objects, Protocols as syntactic wiring $G$, and the full Harness as the Architecture itself. The certificates moved across L1/L2/L3 here are exactly the $\mathrm{Know}$-level objects of that triple; "property preservation under compilation" is the discriminating guarantee — distinct from output-level correctness (which Operon does not claim) and from typed-lambda termination/safety guarantees (cf. Liu's $\lambda_A$, §2 of the paper). Subsequent §2 theorems should be read as compiler-functor instances of preservation under the operon → external-framework adjunction.

**Empirical grounding.** The discriminating claim above — that Operon preserves $\mathrm{Know}$-level structural guarantees but *not* output-level correctness — is not asserted on theory alone. [Banu, *Do Biological Structural Guarantees Earn Their Complexity?*, arXiv:2605.15225](https://arxiv.org/abs/2605.15225) (bib key `banu2026benchmarks`) tests it across 10M+ data points (1,000 trials × 10 seeds) plus a Gemma 4 27B end-to-end run. State-integrity guarantees (DNA repair) are *deterministic* — they hold by construction; behavioral and output-layer guarantees show honest limitations with capable models. The cross-benchmark finding is that biological design earns its complexity through **mechanism-level structural guarantees** (priority gating, signal accumulation with decay, two-signal discrimination), not algorithmic sophistication — and some guarantees are *conditional* (Bayesian stagnation detection delivers 96% vs 2–40% naive only when real-embedding quality is present). This is the empirical answer to *which* of the $\mathrm{Know}$-level certificates transported across L1/L2/L3 actually earn their place; it is also why `operon-langgraph-gates` v0.1.0 cites it for its integrity (§4.1) and real-embedding stagnation (§4.3) gates.

---

## 2. Certificate transport theorems

**Definition (Certificate).** A certificate is a tuple
$$c = (T,\, p,\, \kappa,\, s,\, v)$$
where $T$ is a theorem identifier, $p$ is a parameter dict, $\kappa$ is a conclusion string, $s$ is a source trace identifier, and $v$ is a verify closure resolved via `resolve_verify_fn(T)` from either `_VERIFY_REGISTRY` (dynamic) or `_THEOREM_FN_PATHS` (static) in `operon_ai/core/certificate.py:319`. A certificate's semantic content is the predicate $v(p)$; everything else is transport metadata.

### Theorem 1 (Transport preservation)

Let $F$ be a framework adapter with encode/decode pair $(\phi, \psi)$ such that
$$\psi(\phi(c)).\mathrm{verify}() = c.\mathrm{verify}()$$
for every certificate $c$ whose theorem is registered at both endpoints. Then $F$ preserves certificate validity under composition: for any sequence of transports $F_1 \circ F_2 \circ \cdots \circ F_n$ where each $F_i$ satisfies the round-trip property, the terminal certificate's `verify()` agrees with the initial one.

*Proof sketch.* Induction on $n$, with the base case given by the round-trip hypothesis. The inductive step uses the fact that `verify()` depends only on $(T, p)$ and the local verifier registry — neither is mutated by transport. $\square$

*Witness.* See `TestRoundTrip` in `tests/unit/convergence/test_a2a_certificate.py` — four assertions establish the required equality for the A2A codec $(\phi, \psi) = (\texttt{certificate\_to\_a2a\_part},\, \texttt{certificate\_from\_a2a\_part})$.

### Theorem 2 (Compile-time binding — DSPy, two variants)

Let $\pi = \texttt{dspy.compile}(\pi_0,\, D_{\text{train}},\, m)$ be a compiled DSPy program, and let

$$h_{\text{program}} = \texttt{hash}(\pi_0),\ h_{\text{trainset}} = \texttt{hash}(D_{\text{train}}),\ h_{\text{metric}} = \texttt{hash}(\texttt{source}(m)),\ h_{\text{trace}} = \texttt{hash}(\texttt{traces}(\pi, D_{\text{train}})).$$

**Variant 2a (cheap, shipped 2026-05-01).** The certificate
$$c_{\text{cheap}} = \texttt{Certificate.from\_dspy\_compile}(h_{\text{program}},\, h_{\text{trainset}},\, h_{\text{metric}},\, h_{\text{trace}})$$
asserts the four pinned hashes are recorded. $c_{\text{cheap}}.\mathrm{verify}()$ confirms each hash is well-formed (lowercase hex, length $\geq 8$). Reproducibility itself is checked *downstream*: a later run of `dspy.compile` with the same $(\pi_0,\, D_{\text{train}},\, m,\, \text{seed})$ that produces a different $h_{\text{trace}}$ is the audit signal — model drift, sampling noise, unstable prompt tokenizer, or trainset drift.

**Variant 2b (heavy, deferred).** A hypothetical `Certificate.from_dspy_compile_reproducible(π, D_train, m, h)` would have $c.\mathrm{verify}()$ re-execute compile and check $\texttt{hash}(\texttt{traces}(\pi',\, D_{\text{train}})) = h$ directly. Deferred per §3.1 — re-running compile costs seconds to minutes per `verify()`, which is too expensive for routine verification.

*Status.* Variant 2a (the cheap provenance marker) shipped 2026-05-01, registered as `dspy_compile_pinned_inputs` in `operon_ai.core.certificate._THEOREM_FN_PATHS`. Variant 2b remains future work.

### Theorem 3 (Counterexample-guided convergence — GEPA)

Let GEPA's standard evaluator be $e: \text{Candidate} \to \mathbb{R}$ returning a scalar reward. Replacing $e$ with
$$e_{\text{cert}}: \text{Candidate} \to (\{\top, \bot\},\, \text{Obligation}^*)$$
where obligations are formal unmet proof conditions emitted as free-form text, the mutation step becomes *counterexample-guided*: the reflection LM sees specific unmet obligations rather than a scalar.

**Conjecture.** For theorems with finite obligation sets, GEPA's convergence is bounded by $|\text{Obligations}|$ rather than reward variance.

This is stated as a conjecture, not a theorem: it needs an empirical experiment. The adapter shipped in `operon_ai/convergence/gepa_adapter.py` is precisely the instrument that such an experiment would use.

*Parallel work — topology axis.* Conductor ([Nielsen et al. 2025, ICLR 2026, arXiv:2512.04388](https://arxiv.org/abs/2512.04388)) reaches the same "optimize the agent layer" idea on the topology axis rather than the prompt axis — an RL policy that designs communication graphs (and per-agent prompts) rather than mutating prompts under a fixed graph. Both lines of work treat the agent layer as a learned object; GEPA's reflection LM consumes obligation-shaped feedback, Conductor's policy consumes task reward. The theorem-shape question — whether either converges in a number of mutations bounded by problem structure rather than reward variance — is open for both.

### Theorem 4 (Graceful A2A degradation)

If Operon certificates are encoded as A2A `Part` objects with canonical schema marker `operon.cert.v1`, then any A2A-compliant receiver can
- forward the Part unchanged,
- call `verify()` if and only if the receiver has the matching verify function registered locally, or
- safely ignore the Part if the theorem is unknown.

Therefore certificate emission is non-breaking for legacy A2A agents.

*Proof sketch.* A2A's Part polymorphism permits `DataPart` with arbitrary `data` payloads; the schema marker lets receivers filter. `safe_certificate_from_a2a_part` in `operon_ai/convergence/a2a_certificate.py` implements the "ignore if unknown" branch by catching `UnknownTheoremError` (subclass of `KeyError`) and returning `None`. Test `test_unknown_theorem_safe_returns_none` in `tests/unit/convergence/test_a2a_certificate.py` establishes the witness. $\square$

---

## 3. Integration sections

### 3.1 DSPy — compile-time audit binding

**Repo:** `stanfordnlp/dspy` — 33.9k stars, v3.1.3, Python ≥ 3.10.

DSPy programs (`dspy.Module`, `dspy.Predict`, `dspy.ChainOfThought`) compose into a `Program` that the framework *compiles* via teleprompter/optimizer classes — `BootstrapFewShot`, `MIPRO`, `BootstrapFinetune`, and (since mid-2025) `GEPA`. Compilation consumes a trainset and a user-supplied `metric(example, pred, trace=None) -> float | bool` and emits a new `Program` with optimized few-shot demonstrations and/or prompt text.

The compile boundary — `Program.compile(π₀, D_train, m) → π` — is the natural attachment site for an Operon certificate. The compiled program is an artifact, not a process; at that moment all inputs are pinned and the entire execution trace is reproducible given the seed. A `Certificate` bound here encodes:

- The identity of the uncompiled program (`π₀` structure hash)
- The identity of the training set (`D_train` content hash)
- The identity of the metric function (source-level hash)
- The trace hash `h = hash(traces(π, D_train))` — the reproducibility fingerprint

The shipped public surface (cheap variant 2a) takes pre-computed `sha256` hex digests — *not* Python's built-in `hash()`, which is process-randomized and signed and will not pass the verifier (`length ≥ 8`, lowercase hex). Hashes are computed by the caller from a stable serialization of each artefact:
```python
import hashlib
import inspect
import json

def _digest(payload: bytes) -> str:
    # operon-ai's convention: sha256 lowercase hexdigest; truncation to
    # 8/12/16 chars is permitted (see operon_ai/convergence/memory_bridge.py).
    return hashlib.sha256(payload).hexdigest()

cert = Certificate.from_dspy_compile(
    program_hash=_digest(repr(π_0).encode()),
    trainset_hash=_digest(json.dumps(D_train, sort_keys=True).encode()),
    metric_hash=_digest(inspect.getsource(m).encode()),
    trace_hash=_digest(json.dumps(traces(π, D_train), sort_keys=True).encode()),
)
```
bound to the theorem `dspy_compile_pinned_inputs`, whose verifier confirms each hash is recorded and well-formed. The heavy variant 2b would instead re-run compilation with the same inputs and check that the new trace hash matches — that future hook would carry a different name (e.g. `Certificate.from_dspy_compile_reproducible`) and is deferred.

**Failure modes differ by variant.** Cheap variant 2a's `c.verify()` fails iff the recorded hashes are malformed (missing key, wrong length, non-hex, uppercase) — that is a *recording-integrity* signal, not a drift signal. The cert's role is to pin the four hashes at compile time; **drift is detected downstream**, when a later re-execution of `dspy.compile` with the same inputs produces a different `trace_hash` than the one in the cert. Heavy variant 2b's `c.verify()` would do that re-execution itself, collapsing the recording check and the drift check into one call. Either way, a mismatch (cheap-downstream or heavy-direct) points to the same causes: model drift, sampling noise, unstable prompt tokenizer, or trainset drift.

**Why not ship the heavy variant now?** The full-reproducibility prototype is deferred because (a) re-running compilation is expensive (seconds to minutes), which makes the verifier too slow for routine `verify()` calls, and (b) the composition example in §4 exercises DSPy implicitly via GEPA's `DSPyFullProgramAdapter`, where DSPy serves as the *host* for evolved programs rather than the compile boundary. Adding the heavy verifier is a follow-up PR.

**Cheap variant shipped (2026-05-01).** Per the alternative-framing note below, `dspy_compile_pinned_inputs` is now registered: `Certificate.from_dspy_compile(program_hash, trainset_hash, metric_hash, trace_hash)` builds a provenance certificate whose `verify()` confirms the four hashes are recorded and well-formed (lowercase-hex, length ≥ 8). Code: `operon_ai/convergence/dspy_certificate.py`. The reproducibility witness is downstream — anyone who later re-runs `dspy.compile` with the same inputs and obtains a matching `trace_hash` has confirmed the witness; mismatch is the audit signal. The §8.3 agentflow L2 cert hook (`Certificate.from_agentflow_compile`) shipped same-day, mirroring this binding.

**Alternative framing (now shipped).** A weaker theorem — `dspy_compile_pinned_inputs`, which only checks that the recorded hashes are well-formed and non-empty — was named in this section as the "recommended first step when the full reproducibility theorem arrives." It is now the shipped artefact described in the cheap-variant note above.

### 3.2 GEPA — certificate-based evaluator (Prototype A)

**Repo:** `gepa-ai/gepa` — 3.8k stars, v0.1.1, Python ≥ 3.9.

GEPA is a genetic-Pareto optimizer that evolves candidate text (prompts, code, configs) through LLM reflection over execution feedback. Its adapter protocol (`gepa.core.adapter`) is:

```python
class GEPAAdapter(Generic[DataInst, Trajectory, RolloutOutput]):
    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]: ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...
```

`EvaluationBatch` carries `outputs`, `scores`, and optional `trajectories`. The reflective-dataset pass returns, per mutable component, a sequence of records each containing `Inputs`, `Generated Outputs`, and — critically — `Feedback`.

**Wedge (Theorem 3 instrument).** Replace GEPA's reward-gradient scoring with Operon certificate verification. Pass: the candidate's trajectory yields theorem parameters whose verifier returns `(True, evidence)`. Fail: verifier returns `(False, evidence)`, and the evidence dict is formatted into reflection-ready `Feedback` text that names the theorem, the unmet predicate, and the actual evidence values. The reflection LM receives structured failure reasons instead of a middling scalar — the mechanism by which Theorem 3's conjecture becomes testable.

**Shipping surface** (`operon_ai/convergence/gepa_adapter.py`):

```python
from operon_ai.convergence import (
    OperonCertificateAdapter,
    default_obligation_formatter,
)

def my_harness(candidate, data_inst):
    # Run the candidate on the data instance.
    # Return (output, trajectory, theorem_parameters).
    ...

adapter = OperonCertificateAdapter(
    theorem="behavioral_quality",
    harness=my_harness,
    components=["planner_prompt", "executor_prompt"],
)

# Pass to gepa.optimize(...) — duck-typed, no gepa import here.
```

**Design choices worth calling out:**

1. **No `gepa` import.** The adapter defines its own `EvaluationBatch` dataclass with the same field names. GEPA reads the attributes, not the type. This keeps Operon dependency-free and lets users install `gepa` optionally.
2. **Side-channel verifications.** `evaluate` attaches `_operon_verifications` to the returned batch so `make_reflective_dataset` can consume the verification results without re-running the harness. Dataclass permissiveness makes this clean.
3. **Pluggable formatter.** `default_obligation_formatter` is the baseline; callers with domain-specific obligation languages (e.g., SMT counterexamples, typed errors) can supply their own.
4. **No numeric score gradient.** Scores are strictly `{0.0, 1.0}`. The conjecture predicts that obligation-structured feedback dominates a hand-tuned continuous reward on theorems with finite obligation sets. If that fails empirically, a graded certificate theorem (e.g., `behavioral_quality_graded` returning a ratio) is a natural fallback.

**Seven ready-to-use theorems** from `_THEOREM_FN_PATHS` in `operon_ai/core/certificate.py:211`: `no_false_activation`, `no_oscillation`, `priority_gating`, `state_integrity_verified`, `behavioral_quality`, `behavioral_stability`, `behavioral_stability_windowed`, `behavioral_no_anomaly`. Each ships with a registered verifier; `OperonCertificateAdapter` accepts any of them as `theorem=`.

### 3.3 A2A — Certificate-as-Part (Prototype B)

**Repo:** `a2aproject/A2A` — 23.3k stars, Apache-2.0, Linux Foundation, Google-contributed. The protocol is JSON-RPC 2.0; agents exchange `Message` objects whose `parts` are polymorphic (`TextPart`, `FilePart`, `DataPart`).

**Wedge (Theorem 4 instrument).** Encode Operon certificates as canonical `DataPart` dicts. A2A provides no opinion on semantic correctness — its contracts are identity (`AgentCard`), messaging (`Message`/`Part`), and transport (HTTP+SSE). Certificate-as-Part slots into the contract layer cleanly: emitters declare via AgentCard what theorems they can certify; receivers detect via schema marker; unknown-theorem parts forward or ignore.

**Canonical schema** (`operon_ai/convergence/a2a_certificate.py`):

```json
{
  "kind": "data",
  "data": {
    "schema": "operon.cert.v1",
    "theorem": "<str>",
    "parameters": { ... },
    "conclusion": "<str>",
    "source": "<str>",
    "verification": {
      "holds": <bool>,
      "evidence": { ... }
    } | null
  },
  "metadata": {
    "schema": "operon.cert.v1",
    "mimeType": "application/vnd.operon.cert+json",
    "verifierVersion": "<optional str>"
  }
}
```

**Shipping surface:**
```python
from operon_ai.convergence import (
    certificate_to_a2a_part,
    certificate_from_a2a_part,
    safe_certificate_from_a2a_part,
    agent_card_skill_for_theorem,
    is_certificate_part,
)

# Sender
part = certificate_to_a2a_part(cert, verifier_version="0.36.3")
message = {"parts": [part], "role": "agent"}

# Receiver — strict (raises on unknown theorem)
cert = certificate_from_a2a_part(part)

# Receiver — graceful (returns None on unknown theorem)
cert = safe_certificate_from_a2a_part(part)  # for Theorem 4 degradation

# AgentCard skill declaration
skill = agent_card_skill_for_theorem("behavioral_quality", role="emit")
```

**Design choices worth calling out:**

1. **`DataPart` over `FilePart`.** `DataPart` carries structured JSON directly; `FilePart` would force base64 encoding of what's already JSON-native. `mimeType` lives in `metadata`.
2. **Dual schema marker.** `data.schema` and `metadata.schema` both carry the marker so receivers that strip payloads (e.g., for routing) can still filter on metadata.
3. **Verification embedding is optional.** `certificate_to_a2a_part(cert, verify=False)` forwards a certificate whose verification is deferred. This matters for agents that relay certificates they cannot themselves verify — the receiver can re-verify if it has the verify_fn registered.
4. **`UnknownTheoremError <: KeyError`.** Existing `except KeyError` clauses continue to work; code that wants graceful degradation opts in via `safe_certificate_from_a2a_part`.

**Future work.** A full A2A runtime (spinning up an `a2a-inspector`, registering skills on an AgentCard, exchanging real messages) is out of scope here. The codec is the durable surface; runtime hosting is per-deployment.

---

## 4. Composition example — Operon → GEPA → DSPy → A2A

The following pipeline exercises all four theorems in a single flow:

```
Operon proposer ──► GEPA optimizer ──► DSPy compiled program ──► A2A agent
      [cert]              [cert]              [cert]              [parts]
     Theorem 1           Theorem 3           Theorem 2          Theorem 4
```

1. **Operon proposer** emits candidate prompts paired with *unmet* certificates — a theorem it wants satisfied and a current failure. Certificates serialize via `certificate_to_dict` and travel as annotations in the proposer's output format.
2. **GEPA optimizer** ingests the candidate via `OperonCertificateAdapter.evaluate`. The adapter runs a harness, constructs a certificate per batch item, and returns `(score ∈ {0,1}, trajectories)`. The reflection pass reads obligation text from `_operon_verifications` and feeds it to GEPA's reflection LM. Candidates mutate against *specific unmet obligations* — the Theorem 3 conjecture.
3. **DSPy compiled program.** GEPA's built-in `DSPyFullProgramAdapter` evolves a `dspy.Module`. At compile completion, `Certificate.from_dspy_compile` (Theorem 2 cheap variant 2a, shipped 2026-05-01) binds a provenance cert recording the four pinned hashes. The heavy reproducibility verifier (variant 2b) remains deferred.
4. **A2A agent hosting.** The compiled program is wrapped as an A2A agent. Every response message includes certificate Parts — one for the GEPA-evolved behavioral theorem, one for the DSPy compile-provenance cert (Theorem 2 cheap variant). Downstream agents that share the theorem registry can re-verify; others forward or ignore (Theorem 4).
5. **Theorem 1** governs the entire chain: because each hop's encode/decode pair preserves `verify()`, the terminal A2A message's certificate agrees with the original proposer's claim. Audit is preserved across vendors.

This diagram is the main asset of this memo for paper-adjacent reuse: it unifies Operon's proposer interface, GEPA's optimizer, DSPy's artifact layer, and A2A's protocol into a single provenance chain, gated on certificate-valued boundaries at every hop.

**Categorical reading.** Each arrow above is a compiler functor in the sense of [Banu, *Harness Engineering as Categorical Architecture*, arXiv:2605.12239](https://arxiv.org/abs/2605.12239) (`banu2026harness`); the chain commutes on the $\mathrm{Know}$-component by construction (Theorem 1 is the preservation lemma; Theorems 2–4 specialise the lemma at the artifact, optimizer, and protocol surfaces). What this composition does not claim is output equivalence: the harness preserves recorded structural guarantees, not model behaviour. The paper's §4 (preservation) is the closest formal companion to this composition example.

---

## 5. What ships with this memo

| Artifact | Path | Purpose |
|---|---|---|
| Memo (this doc) | `docs/site/external-frameworks.md` | L1/L2/L3 taxonomy + 4 theorems |
| GEPA adapter | `operon_ai/convergence/gepa_adapter.py` | Prototype A (Theorem 3 instrument) |
| A2A codec | `operon_ai/convergence/a2a_certificate.py` | Prototype B (Theorem 4 instrument) |
| GEPA tests | `tests/unit/convergence/test_gepa_adapter.py` | 14 tests, adapter contract |
| A2A tests | `tests/unit/convergence/test_a2a_certificate.py` | 27 tests, round-trip + degradation |

All 41 tests pass under `pytest tests/unit/convergence/test_gepa_adapter.py tests/unit/convergence/test_a2a_certificate.py`.

## 6. Explicitly out of scope

- **OpenMythos** (`kyegomez/OpenMythos`) — model architecture, not an agent framework. Could appear as an `LLMProvider` backend, identical to any other model wrapper.
- **Swarms gate injection** — swarms is already integrated bidirectionally at L1; deepening (e.g., injecting `StagnationGate`/`IntegrityGate` at `GraphWorkflow` edges) is a separate PR.
- **Heavy DSPy reproducibility verifier** — Theorem 2 *cheap variant* (`dspy_compile_pinned_inputs`) shipped 2026-05-01; the heavy variant that re-executes compile and checks trace-hash equality remains deferred per §3.1's cost argument.
- **A2A runtime hosting** — codec ships; spinning up an A2A server with `a2a-inspector` is per-deployment.
- **LLM output validation** — Operon validates *process* (gates, certificates), not *output content*. No LLM output validation for toxicity, schema conformance, or hallucination filtering ships from this codebase; data-layer frameworks such as Guardrails AI are the sibling layer for that surface (see §8.2).

## 7. Follow-up tasks

1. **Empirical Theorem 3.** Build a GEPA-vs-reward-gradient benchmark on a bio theorem (`behavioral_stability_windowed`) and measure mutations-to-convergence. Target: 2× reduction on the certificate side, or refute the conjecture cleanly.
2. **Lightweight DSPy theorem.** Ship `dspy_compile_pinned_inputs` — the cheap provenance variant of Theorem 2 — before the full reproducibility verifier.
3. **A2A runtime smoke test.** Stand up a minimal A2A server exposing an Operon-certified skill; run `a2a-inspector` against it; verify the certificate Part round-trips through the real wire format.
4. **`gepa_candidate_improvement` theorem.** Register a verifier that certifies a GEPA-evolved candidate dominates its parent on the Pareto frontier; close the outer loop.

## 8. Landscape addenda (2026-04-24)

Triage pass on three repos surfaced as possible convergence targets. Placed against the L1/L2/L3 taxonomy of §1 with a verdict — no code shipped in this initial triage; subsequent updates (below) record code that has since landed off the verdicts.

**Update (2026-04-30):** the `operon-langgraph-gates` wedge that gated §8.1 and §8.3 below has shipped as `v0.1.0` ([release](https://github.com/coredipper/operon-langgraph-gates/releases/tag/v0.1.0), [PyPI](https://pypi.org/project/operon-langgraph-gates/0.1.0/)). The queue positions and "deferred" verdicts have been updated to reflect post-ship state; original analysis is preserved in git history (commit `375aedb`).

**Update (2026-05-01):** the §8.1 follow-up has shipped. `operon_ai.convergence.GascityCertificateAdapter` lands in-tree alongside the existing GEPA / Swarms / DeerFlow / Ralph / Scion / A-Evolve adapters, with a dogfood test that consumes `STAGNATION_THEOREM` and `INTEGRITY_THEOREM` directly from `operon-langgraph-gates` v0.1.0. agentflow (§8.3) is promoted to queue position 1.

**Update (2026-05-01, later):** the §2 cheap-variant T2 binding (`Certificate.from_dspy_compile`) and the §8.3 L2 agentflow cert hook (`Certificate.from_agentflow_compile`) ship together as a coherent pair. Both are provenance markers — they record the pinned-input hashes at compile time but do not re-run compilation. See §2 *Status* and §8.3 *Verdict* for details.

**Update (2026-05-11):** §8.4 added — `veloryn-xray`, surfaced via a [2026-05-10 comment](https://github.com/langchain-ai/langgraph/issues/6731#issuecomment-4415887497) on the same LangGraph stagnation issue that motivated `operon-langgraph-gates` v0.1.0. Recorded as the observational-only baseline; the contrast clarifies what runtime gating buys over post-hoc analysis.

**Update (2026-05-11, later):** §8.3 L1 adapter has shipped. `operon_ai.convergence.AgentflowCertificateAdapter` lands in-tree alongside the existing gascity / Swarms / DeerFlow / Ralph / Scion / A-Evolve adapters; three event mirrors (`NodeEvent`, `EdgeEvent`, `EvolveEvent`) cover agentflow's runtime-observable boundaries since the framework exposes no hook surface of its own. Both wedge angles for agentflow (L1 runtime, L2 compile-time) are now closed; agentflow is the first framework in §1 with both layers shipped in-tree.

**Update (2026-05-12):** §8.6 added — *The Memory Curse* (Liu et al., [arXiv:2605.08060](https://arxiv.org/abs/2605.08060)) frames the empirical failure mode for multi-agent cooperation under expanded context windows. Recorded as a scope-line addendum: the paper supplies the failure-mode characterization, operon's existing memory layer provides the surfaces where sanitization could live, no mitigation shipped this turn.

**Update (2026-05-12, later):** §8.7 added — *Agentic-imodels* (Singh et al., [arXiv:2605.03808](https://arxiv.org/abs/2605.03808)) is the first method paper in the operon corpus claiming evolution-as-optimization yields generalizing, agent-readable artefacts (within-domain across tabular datasets; +73% BLADE on Copilot CLI and Claude Code). Method-peer to §8.5. Recorded as a methodological-evidence addendum; the operon-internal C8 finding ("abstractions generalize as code structure, not optimization") frames a within-domain test as the open follow-up.

**Update (2026-05-13):** the empirical companion to `banu2026harness` is now on arXiv — [Banu, *Do Biological Structural Guarantees Earn Their Complexity?*, arXiv:2605.15225](https://arxiv.org/abs/2605.15225) (bib key `banu2026benchmarks`). It supplies the measured grounding for the §1 *Empirical grounding* paragraph: integrity (state-integrity / DNA repair) certificates are deterministic, behavioral/output-layer guarantees show honest limitations with capable models, and the real-embedding stagnation guarantee is conditional on embedding quality (96% vs 2–40% naive). This closes the citation graph for the second of the two paper-corpus self-references (`banu2026harness` closed 2026-05-13 via operon PR #153).

**Update (2026-05-19):** §8.4 extended with an *adoption read* — the v0.1→v0.2 decision for `operon-langgraph-gates` was gated on external signal; the signal (18–19 days post-release) is weak pull, recorded honestly here in the SWE-bench-null style. v0.2 scope decided as *harden + dogfood contract*, with new primitives explicitly pull-gated. No code shipped.

**Update (2026-05-26):** §8.8 added — *The Log is the Agent* (Nakajima, [arXiv:2605.21997](https://arxiv.org/abs/2605.21997), bib key `nakajima2026activegraph`) frames ActiveGraph as an event-sourced **alternative L1 substrate**, sibling of LangGraph itself rather than of the adapter family. Recorded as a *deferred-by-discipline* attach-point candidate: ActiveGraph's `@behavior()` / `@relation_behavior()` decorators are natural gate attach points, but no adapter ships until a pull trigger fires (per the §8.4 v0.2 scope decision). The entry strengthens the portability claim by adding a fifth substrate the gates could attach to (LangGraph in-tree, gascity §8.1, agentflow §8.3, ActiveGraph deferred) without growing operon's API surface. No code shipped.

### 8.1 gascity — L1 coordination runtime, adapter shipped

Gas City (`gastownhall/gascity`, v1.0 released 2026-04-21) is Steve Yegge's Go-based coordination runtime for durable, long-lived agent teams. Core primitives are **Packs** (declarative module bundles), **Beads** (two-level work-routing abstraction with formulas/molecules/waits as first-class), **MEOW** (versioned knowledge graphs), and persistent agents running in tmux sessions with git-versioned Dolt audit trails. It ships no validation surface of its own — `internal/validation/` covers config schema only — but exposes rich lifecycle hooks (`SessionStart`, `PreToolUse`, `UserPromptSubmit`, `Stop`) plus a four-tier integration model (JSON preset → settings hook → plugin → non-interactive).

**Placement.** L1, same layer as the existing Swarms / DeerFlow / AnimaWorks / Ralph / A-Evolve / Scion adapters and the LangGraph guarded-graph compiler. Natural gate attach points: `hooks/` (per-turn invariant re-validation, akin to pre-guard), `dispatch/` (pre-nudge structural check), `mail/` (message-boundary certificates). Certificates would emit into the Beads/Dolt audit trail — an unusually strong home for compile-time artefacts, since Dolt gives versioned queryable history for free.

**Verdict — adapter shipped (2026-05-01).** Gascity's hook surface remains the cleanest gate attach point of any L1 framework surveyed to date. The duck-typed adapter — `operon_ai.convergence.GascityCertificateAdapter` at `operon_ai/convergence/gascity_adapter.py` — mirrors `gepa_adapter.py`. Three event mirrors (`HookEvent`, `DispatchEvent`, `MailEvent`) cover gascity's `hooks/`, `dispatch/`, and `mail/` integration points; `verification_to_dolt_envelope()` renders certificates as flat JSON dicts for the Beads/Dolt audit trail. The adapter takes theorem names as constructor parameters — callers pass either operon-ai built-ins or the public constants `STAGNATION_THEOREM` (`"behavioral_stability_windowed"`) and `INTEGRITY_THEOREM` (`"langgraph_state_integrity"`) imported from `operon-langgraph-gates` v0.1.0. A skipped-by-default dogfood test exercises the latter path; it confirms the v0.1.0 stable surface is consumable from a sibling adapter without binding to upstream string literals.

### 8.2 Guardrails AI — complement, not wedge

Guardrails AI (`guardrails-ai/guardrails`) is a mature ($7.5M seed, ~2.9k stars, production-adopted) output-validation framework. Primitives are `Validator` (e.g., `toxic_language`, `competitor_check`, schema coercion), composed into a `Guard` via `.use()`, with `OnFailAction` policies (`REASK` / `FIX` / `FILTER` / `REFRAIN` / `EXCEPTION`) determining behaviour on validation failure. Ships as Python SDK, as a Flask-based server with OpenAI-compatible endpoints, and as a plug-in to LiteLLM proxy and Cloudflare AI Gateway. Validation is best-effort, post-hoc, runtime — there is no certificate or compile-time notion.

**Placement.** Below L1 — same level as OpenMythos in §1. Guardrails validates *what data flows through an agent node* (shape, toxicity, PII, schema); Operon gates validate *how the graph re-shapes under load* (stagnation, contradiction, invariant preservation). The two layers compose cleanly: inside a guarded LangGraph, Guard runs *inside* each agent node (data boundary), Gate runs *between* nodes (process boundary).

**Verdict — complement, not wedge.** Guardrails covers the exact territory §6 marks as explicitly out of scope for Operon ("No LLM output validation for toxicity, schema conformance, or hallucination filtering"). This addendum reaffirms that scope line: Operon should not build output validators, and does not need a Guardrails adapter to claim coverage of that layer — pointing at Guardrails as the sibling framework is the right answer. A joint-demo snippet (one guarded LangGraph node wrapped by both a `Guard` and an Operon pre/post-guard) is the only artefact worth considering, and only if a user asks.

### 8.3 agentflow — L1 adapter + L2 cert hook, both shipped

BeraBuilds AgentFlow (`berabuddies/agentflow`, ~1.1k stars, active Apr 2026, no tagged releases) is a Python DAG orchestrator for codex/claude/kimi agents. Primitives are `Graph()` context manager, named agent nodes, operators `>>` (sequence), `fanout` + `merge` (parallel), `on_failure` (loops), `max_iterations` (bounded recursion), and Jinja2 output interpolation. Distinctive secondary feature: a built-in **agent evolution / tuning pipeline** — `agentflow evolve` compiles successful traces into tuned agent versions written to `.agentflow/tuned_agents/`. Ships as Python library + CLI; no plugin surface, no middleware, no config serialization layer (topology is pure Python).

**Placement.** Primary **L1 topology** (`Graph` is a DAG of agent calls — same shape as Swarms/DeerFlow/LangGraph/gascity). Secondary **L2 artefact** — the tuned-agent output is a frozen-prompt-equivalent artefact, landing in the same territory as DSPy compile (§2 Theorem 2) and GEPA optimization (§2 Theorem 3). This is the first L1 framework surveyed with a native L2 evolution loop baked in.

**Verdict — both wedge angles shipped.**
1. **L1 adapter — shipped (2026-05-11).** `operon_ai.convergence.AgentflowCertificateAdapter` at `operon_ai/convergence/agentflow_adapter.py` mirrors `gascity_adapter.py` shape. Three event mirrors (`NodeEvent`, `EdgeEvent`, `EvolveEvent`) cover agentflow's runtime-observable boundaries — pre/post node execution, inter-node transitions, and the `evolve` compile boundary — since agentflow exposes no hook surface of its own. Callers wrap `Graph.run()` in a façade that emits these events to user-supplied harnesses; the adapter takes theorem names as constructor parameters and re-uses `verification_to_dolt_envelope()` from the gascity sibling for Beads/Dolt-friendly JSON. Includes a skipped-by-default dogfood test that imports `STAGNATION_THEOREM` / `INTEGRITY_THEOREM` from `operon-langgraph-gates` v0.1.0 and confirms the L1 adapter consumes them at agentflow attach points.
2. **L2 certificate hook for `evolve` — shipped (2026-05-01).** `Certificate.from_agentflow_compile(graph_hash, traces_hash, tuned_agent_hash)` registers a `agentflow_evolve_pinned_inputs` provenance certificate, mirroring the §2 cheap-variant T2 DSPy binding (which shipped the same day). The verifier checks that the three pinned hashes — uncompiled `Graph()`, input traces, and tuned-agent output — are recorded and well-formed; reproducibility itself is checked downstream by re-running `agentflow evolve` and comparing the tuned-agent hash. Code: `operon_ai/convergence/agentflow_certificate.py`. The L1 adapter now also exposes this theorem at the runtime side via `AgentflowCertificateAdapter(theorem="agentflow_evolve_pinned_inputs", harness_evolve=...)` — pinning the same three hashes from the `EvolveEvent` payload. A heavy variant that re-runs `evolve` and checks for hash equality is the natural follow-up but not yet scoped.

The L2 angle remains the distinctive play — no other L1 framework in §1 ships its own optimizer. The L1 adapter brings agentflow into structural parity with gascity at the topology layer; together they make agentflow the first framework in the §1 table to have both runtime gate attach points (L1) and a compile-time provenance hook (L2) shipped in-tree.

### 8.4 veloryn-xray — observational-only baseline

`veloryn-intel/veloryn-xray` ([repo](https://github.com/veloryn-intel/veloryn-xray), surfaced 2026-05-10 via a [comment on LangGraph issue #6731](https://github.com/langchain-ai/langgraph/issues/6731#issuecomment-4415887497) — the same stagnation thread that motivated `operon-langgraph-gates` v0.1.0) is a Python trace-analysis tool with a LangChain callback adapter. Public surface is a single `XRayCallbackHandler` plus an `analyze()` method that returns a `to_dict()` report after the chain has run. From the comment's own description: "The callback is observational only. It captures outputs and replays the trace afterward through deterministic execution analysis."

**Placement.** Sibling of `operon-langgraph-gates` at the LangChain/LangGraph boundary — both target the *behavioral-stability* problem named in issue #6731 — but at opposite ends of the time axis. X-Ray runs *after* execution; `StagnationGate` runs *during* execution.

**Verdict — observational baseline, not a wedge.** The distinction is the load-bearing claim of the operon-langgraph-gates wedge: a callback can *report* that a workflow stopped materially changing, but cannot *prevent* the wasted tokens, retries, or terminal cost between the stagnation onset and the eventual `recursion_limit` exit. `StagnationGate` ends the cycle at the boundary where behavioral similarity is first detected, and the certificate it emits is a *runtime artefact* — pinned at the gate event, signed by the verifier registry, replayable through the audit trail. X-Ray's analysis is a *post-hoc artefact* — synthesised after the trace closes, with no structural authority over execution. Both can coexist on the same graph; pairing the two would let X-Ray annotate *why* a gate fired, while the gate keeps the runtime cost bounded.

No code shipped with this addendum — landscape entry only. The wedge claim stands: certificates that carry *structural authority* (refuse to continue) are categorically distinct from analyses that carry *diagnostic authority* (explain after the fact), and operon-langgraph-gates remains the only framework surveyed that gates LangGraph execution at the cycle boundary with replayable certificate evidence.

**Adoption read & v0.2 scope (2026-05-19).** Recorded in the same honest style as the [SWE-bench Phase 2 null result](https://banu.be/blog/openhands-gates-swebench-lite-null/): the v0.1→v0.2 gate was explicitly "decide after external signal," so the signal is reported whichever way it points. At 18–19 days post-v0.1.0: 1 GitHub star, **0 issues, 0 PRs, 0 page-views**; 73 clones / 25 unique cloners (curiosity, no engagement); PyPI download count unavailable. Issue #6731 is *closed*, and the most recent comment on it is veloryn-xray's (2026-05-10) — the observational-only sibling now holds the visible surface on the thread that motivated the wedge. The structural-vs-diagnostic differentiation is intact; discovery and demonstrated pull are not. The honest reading is **weak external pull**, not a refuted thesis (the evidence is non-trial, not trial-and-failure — so neither "declare done" nor a marketing push is warranted yet). **v0.2 scope is therefore *harden + dogfood contract*:** promote the skipped sibling-adapter dogfood test (gascity / agentflow consuming `STAGNATION_THEOREM` / `INTEGRITY_THEOREM`) to a supported cross-package contract test and document that consumption path as the portability proof artefact — no new API surface. A `QuorumGate` (Paper 4 quorum-sensing benchmark) is the only disciplined future primitive and is **pull-gated**: deferred until a first external issue, a sibling adapter that needs it, or a measured download/clone inflection. Speculative primitive growth with no pull is the BUDGET a-priori-pick error the 2026-04-21 survey already corrected once.

**v0.2 execution (2026-05-19).** Tasks 1–2 shipped in `operon-ai` PR #182: the sibling-adapter dogfood (`TestOperonLanggraphGatesDogfood` in `tests/unit/convergence/test_gascity_adapter.py` and `test_agentflow_adapter.py`) is now a *real* CI contract — `operon-langgraph-gates` is a `dev` extra so the test executes instead of `importorskip`-skipping, and each case asserts the full replayable-evidence round-trip (`verification_to_dolt_envelope` JSON render + `certificate_to_dict` → `certificate_from_dict` → `verify()` agreement, plus `resolve_verify_fn` registry resolution for the integrity theorem). The consumer-side contract is documented in [`operon-langgraph-gates` README → *Certificate theorem name and verification*](https://github.com/coredipper/operon-langgraph-gates#certificate-theorem-name-and-verification). Task 4 (package hardening) was found already complete in the `0.1.x` line — `py.typed`, explicit `__all__`, `tests/test_public_api.py` stability assertions, bounded `operon-ai` / `langgraph` pins — so **no `operon-langgraph-gates` release follows from v0.2**: nothing in that package changed (the work landed in `operon-ai`), and a hollow `0.2.0` would violate the package's own "0.2.0 = breaking changes" SemVer policy. The wedge's portability claim is now *executed*, not merely asserted.

### 8.5 Reinforced Agent — academic validation of the runtime-gating wedge

Ta et al., *Reinforced Agent: Inference-Time Feedback for Tool-Calling Agents* (2026-04-29, [arXiv:2604.27233](https://arxiv.org/abs/2604.27233)) shifts tool-call evaluation from post-hoc trajectory assessment to **inference-time review**: a specialized reviewer agent evaluates provisional tool calls before execution, mirroring operon's `reviewer_gate` pattern (`operon_ai/patterns/review.py`). The paper introduces **Helpfulness/Harmfulness** metrics quantifying the tradeoff between corrected errors and newly-introduced ones — answering, in the framing of §8.4, whether a runtime gate is net-positive at a population level rather than just structurally available.

**Placement.** Same boundary as §8.4 (LangChain/LangGraph runtime, behavioral guard), but published evidence rather than a competing tool. The reviewer-agent pattern is operon's existing `reviewer_gate`; the H/H metric is the missing measurement layer over it.

**Verdict — validates the wedge.** The paper supplies measured benefit-to-risk ratios for the runtime-review architecture: 3:1 for o3-mini and 2.1:1 for GPT-4o on BFCL irrelevance detection; +7.1% on multi-turn Tau2-Bench; GEPA prompt optimization adds an additional +1.5–2.8%. These numbers are the kind of evidence the §8.4 framing claimed without citing — that *structural authority* (refuse to continue) is empirically net-positive over *diagnostic authority* (post-hoc analysis), at least within the regime measured by these benchmarks. The choice of reviewer model and prompt is shown to be the load-bearing knob; the same paper's GEPA result lines up with operon's §2 Theorem 3 conjecture on the prompt axis.

**What ships in operon.** An H/H metric utility lands in [`eval/patterns/reviewer_gate_hh_metric.py`](../../eval/patterns/reviewer_gate_hh_metric.py) so operon users can compute the same Helpfulness/Harmfulness population statistic over any `reviewer_gate`-instrumented harness — taking two lists of `TrajectoryOutcome(task_id, correct, intervened)` records (one base, one reviewed) and returning a `HelpfulnessHarmfulnessMetric` with the corrected/degraded counts and the benefit-to-risk ratio. No new theorem registration in this pass; registering a `reviewer_gate_net_positive` cheap-variant theorem on top of H/H is the natural follow-up if a Certificate-shaped artifact is wanted.

### 8.6 Memory Curse — empirical scope-line for multi-agent cooperation

Liu et al., *The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents* (2026-05-08, [arXiv:2605.08060](https://arxiv.org/abs/2605.08060)) report that across 7 LLMs × 4 social-dilemma games × 500 rounds, expanding the context window *degrades* cooperation in 18 of 28 model–game settings. Three findings define the failure mode:

1. **Mechanism.** Lexical analysis of 378,000 reasoning traces shows the cooperation collapse is driven by erosion of *forward-looking intent*, not rising paranoia. A LoRA adapter trained exclusively on forward-looking traces mitigates the decay and transfers zero-shot to different games.
2. **Content, not length.** Holding prompt length fixed while replacing visible history with synthetic cooperative records *restores* cooperation. The trigger is memory **content**, not size.
3. **Deliberation amplifies the collapse.** Ablating explicit Chain-of-Thought reasoning often *reduces* the memory curse — extended deliberation makes things worse, not better.

**Placement.** Cross-cuts the §1 L1 row (every multi-agent topology — `QuorumSensing`, `Cascade`, `specialist_swarm`, anything iterated over `BioAgent` collections — is in scope) and the §2 reasoning layer (the CoT-amplifies finding bears on every reviewer-gated and reflective architecture, including `reviewer_gate` and the GEPA reflection loop).

**Verdict — scope line, no shipped mitigation.** This paper supplies what was previously implicit: an empirical failure-mode characterization for operon's multi-agent surface. The relevant operon-side artefacts are *adjacent*, not solutions:

- `operon_ai/state/histone.py` ships `HistoneStore` with per-marker decay (`decay_hours` per `EpigeneticMarker`) — a decay schedule over memory content, not a curated-content gate.
- `operon_ai/memory/episodic.py` ships `EpisodicMemory` with tiered retention (`WORKING` / `EPISODIC` / `LONGTERM`) and a `DECAY_RATES` table — tier policy is age-based, not cooperation-aware.
- `operon_ai/memory/bitemporal.py` ships `BiTemporalMemory.correct_fact(...)` — overwrites stale facts but does not synthesise cooperative records.
- `operon_ai/organelles/lysosome.py` ships `Lysosome` waste digestion (`WasteType`, `Waste`, `DigestResult`) — capacity-driven, not content-driven.
- `operon_ai/core/denature.py` ships `NormalizeFilter`, `SummarizeFilter`, `StripMarkupFilter`, and `ChainFilter` — generic text rewriters that could in principle implement a sanitization pipeline.

Each is a *surface where sanitization-at-gate-boundary could be implemented*; none currently is, and the biological metaphors they spring from (decay, autophagy, denaturation) are coincidentally well-shaped, not causally responsive to this finding. A future *memory hygiene* certificate that pairs a synthesised-cooperative-history fixture with a verified call site would be the natural follow-up artefact; explicitly out of scope for this addendum.

**The operationally important claim for operon is the third.** Deliberation amplifies the collapse. Reviewer-gated patterns (`operon_ai/patterns/review.py`) and the GEPA reflection loop are deliberation-heavy; if they are deployed in iterated multi-agent settings, they may exhibit the memory curse more strongly than non-deliberative baselines. The §8.5 Helpfulness/Harmfulness metric measures reviewer net-positivity at the population level over independent trajectories; an iterated-cooperation analogue — forward-looking-intent ratio over a multi-round trajectory log — is the natural §8.6 follow-up benchmark.

No code shipped with this addendum — landscape entry only. The wedge claim is that operon's structural-authority gates (refuse to continue) are categorically distinct from observational tools; the Memory Curse paper's contribution is to mark a regime where even structural-authority gates may need a *memory-content* gate alongside the structural one to preserve cooperation properties under iterated play.

### 8.7 Agentic-imodels — autoresearch evolution for agent-facing interpretability

Singh et al., *Agentic-imodels: Evolving agentic interpretability tools via autoresearch* (2026-05-05, [arXiv:2605.03808](https://arxiv.org/abs/2605.03808)) propose an autoresearch loop that evolves scikit-learn-compatible regressors for tabular data, jointly optimizing predictive accuracy and a novel LLM-based **simulatability** metric — whether an LLM can predict model behavior from the model's string representation alone. Three claims define the method:

1. **Joint objective is realizable.** Evolved regressors improve both accuracy and simulatability simultaneously; the two objectives are not strict trade-offs at the scale tested.
2. **Within-domain generalization.** Models evolved on one tabular dataset transfer to other tabular datasets without re-evolution.
3. **Downstream agentic boost.** Evolved models lift BLADE benchmark performance by up to **73% for Copilot CLI and Claude Code** — the first reported instance in the operon corpus where evolution-for-agent-readability moves a downstream agentic-data-science score by a margin of that size.

**Placement.** Method-peer to §8.5 (Reinforced Agent / H/H metric). Both papers shift evaluation upstream: §8.5 moves tool-call assessment from post-hoc trajectory to inference-time review; §8.7 moves model interpretability from post-hoc explanation to evolution-time objective. Both targets are reviewer-style surrogates over LLM judgement rather than human-readability proxies.

**Verdict — method paper, adjacent surfaces, open within-domain test.** The paper supplies the first concrete positive result in the operon corpus for *evolution as a generalization mechanism over agent-readable artefacts*. The operon-side surfaces that connect are:

- The **autoresearch skill** implements the workflow-level loop (try → measure → keep → discard) at the orchestration layer, not at the model-internal layer Singh evolves over.
- [`operon_ai/convergence/gepa_adapter.py`](../../operon_ai/convergence/gepa_adapter.py) ships `GEPAAdapter` bridging operon verification certificates into GEPA's reflective-evolution optimizer — the closest in-tree analog to Singh's autoresearch loop, but the optimization target is prompt/topology, not a regressor's string representation.
- The §8.5 H/H metric ([`eval/patterns/reviewer_gate_hh_metric.py`](../../eval/patterns/reviewer_gate_hh_metric.py)) and the absence of any **simulatability** scoring surface in `operon_ai/` together define the gap: operon measures reviewer net-positivity at the population level, not whether an agent can predict an artefact's behavior from its source representation at the per-artefact level.

**C8 tension — different scopes, not a refutation.** An operon-internal study concluded that evolutionary optimization over **graph topologies** *degraded* generalization across structural types (LLM-proposer 0.49 → 0.36; Tournament 0.44 → 0.60 was the exception). Singh reports *within-domain* generalization across **tabular datasets** under a joint accuracy + simulatability objective. The two findings are not in direct contradiction: the sample spaces are disjoint (cross-structural-type vs within-domain), the optimization targets differ (graph topology vs regressor source representation), and the joint-objective signal Singh exploits (simulatability) was absent in C8. The honest position is that **Singh defines an experiment operon has not yet run**: a within-domain autoresearch loop over an operon-internal regressor or reviewer-policy surface, scored by an LLM simulatability probe. Until that loop is run, Singh's positive result is an *open hypothesis* about operon's autoresearch surface, not a settled method.

**The operationally important claim for operon is the third.** A 73% BLADE boost specifically named for Claude Code — the same tool operon users are running today — is a concrete signal that this paper's downstream domain overlaps the operon user surface. A within-domain replication targeting `reviewer_gate` prompt evolution scored by a simulatability probe over the resulting prompt is the natural follow-up artefact; explicitly out of scope for this addendum.

No code shipped with this addendum — landscape entry only. The wedge claim is that operon's structural-authority gates (refuse to continue) and Singh's evolved interpretability artefacts (designed to be simulatable) compose at the agent surface: a reviewer-gate's *what fired* signal pairs naturally with an evolved-artefact *why this artefact* signal, when both are scored against an LLM probe rather than a human-readability proxy.

### 8.8 ActiveGraph — alternative L1 substrate, attach-point candidate (deferred)

Nakajima, *The Log is the Agent: Event-Sourced Reactive Graphs for Auditable, Forkable Agentic Systems* (2026-05-21, [arXiv:2605.21997](https://arxiv.org/abs/2605.21997); product surface at [activegraph.ai](https://activegraph.ai/)) inverts the conventional agent-framework layering: the append-only event log is the source of truth, the working graph is a deterministic projection of that log, and behaviors — ordinary functions, classes, LLM-backed routines, or `@relation_behavior()` logic attached to typed edges — react to graph changes and emit new events. No component instructs another; coordination happens entirely through the shared graph. Three named runtime properties: **deterministic replay** of any run from its log, **cheap forking** (`runtime.fork(at_event=...)`, "shared prefix replays from cache"), and **end-to-end lineage** from a high-level goal down to the individual model call that produced each artifact. One formal guarantee: a *"determinism contract that makes replay sound."* Sits explicitly in the **BabyAGI lineage**.

**Placement.** L1 *substrate* — sibling of LangGraph itself, **not** of the L1 *adapter* row in §1. Where the §1 adapters (Swarms / DeerFlow / gascity / agentflow / LangGraph guarded-graph) treat topology as given and emit certificates against it, ActiveGraph is *another conversation-loop framework whose topology is dynamic by construction*. The natural gate attach points are `@behavior()` (akin to a LangGraph node hook) and `@relation_behavior()` on typed edges (the process-boundary territory `StagnationGate` and `IntegrityGate` are defined for, distinct from Guardrails' data-boundary territory in §8.2). ActiveGraph's fork+replay machinery is unusually well-shaped for certificate replay over forked event prefixes — a property no L1 framework surveyed to date supplies natively.

**Verdict — alternative substrate, attach-point candidate, no adapter scheduled.** ActiveGraph's existence is **corroborating evidence** for the portability claim that motivates the operon-langgraph-gates wedge: *harness-level structural certificates are model-independent and framework-portable*. Through the Ning et al. *Code as Agent Harness* survey lens (arXiv:2605.18747, bib key `ning2026codeharness`), ActiveGraph delivers §5.2.1 metric dimensions **(i) trajectory efficiency, (iv) state consistency, and (vi) replayability** solidly. It does **not** deliver §5.2.2 — *"the central missing abstraction is a verification stack with explicit scope … each artifact should declare what it verifies, what it cannot verify, and what confidence it provides"* — there is no scope-typed certificate, no declarative "this verifies X but not Y" surface, no machine-checked binding spec analogous to `operon-langgraph-gates/specs/certificate-binding.allium`. The §5.2.2 gap is the same one operon's v0.2 dogfood contract (§8.4 *v0.2 execution*) was hardened to demonstrate.

Per the v0.2 scope discipline (the 2026-05-19 decision recorded in §8.4 above), **no `ActiveGraphCertificateAdapter` ships until a pull trigger fires** — a first external issue, a sibling adapter that genuinely needs it, or measured download/clone inflection on the existing surface. Adding ActiveGraph to the §8 landscape *strengthens the portability claim without growing the API* — the same arithmetic that made the gascity (§8.1) and agentflow (§8.3) entries load-bearing once shipped, applied here in reverse: the substrate is recorded, the adapter waits. Honest position: ActiveGraph is the strongest external corroboration of the wedge-claim to date precisely because it solves a different layer (substrate, not verifier) with the same "log is the audit trail" instinct.

No code shipped with this addendum — landscape entry only.
