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

### 8.1 gascity — L1 coordination runtime, adapter shipped

Gas City (`gastownhall/gascity`, v1.0 released 2026-04-21) is Steve Yegge's Go-based coordination runtime for durable, long-lived agent teams. Core primitives are **Packs** (declarative module bundles), **Beads** (two-level work-routing abstraction with formulas/molecules/waits as first-class), **MEOW** (versioned knowledge graphs), and persistent agents running in tmux sessions with git-versioned Dolt audit trails. It ships no validation surface of its own — `internal/validation/` covers config schema only — but exposes rich lifecycle hooks (`SessionStart`, `PreToolUse`, `UserPromptSubmit`, `Stop`) plus a four-tier integration model (JSON preset → settings hook → plugin → non-interactive).

**Placement.** L1, same layer as the existing Swarms / DeerFlow / AnimaWorks / Ralph / A-Evolve / Scion adapters and the LangGraph guarded-graph compiler. Natural gate attach points: `hooks/` (per-turn invariant re-validation, akin to pre-guard), `dispatch/` (pre-nudge structural check), `mail/` (message-boundary certificates). Certificates would emit into the Beads/Dolt audit trail — an unusually strong home for compile-time artefacts, since Dolt gives versioned queryable history for free.

**Verdict — adapter shipped (2026-05-01).** Gascity's hook surface remains the cleanest gate attach point of any L1 framework surveyed to date. The duck-typed adapter — `operon_ai.convergence.GascityCertificateAdapter` at `operon_ai/convergence/gascity_adapter.py` — mirrors `gepa_adapter.py`. Three event mirrors (`HookEvent`, `DispatchEvent`, `MailEvent`) cover gascity's `hooks/`, `dispatch/`, and `mail/` integration points; `verification_to_dolt_envelope()` renders certificates as flat JSON dicts for the Beads/Dolt audit trail. The adapter takes theorem names as constructor parameters — callers pass either operon-ai built-ins or the public constants `STAGNATION_THEOREM` (`"behavioral_stability_windowed"`) and `INTEGRITY_THEOREM` (`"langgraph_state_integrity"`) imported from `operon-langgraph-gates` v0.1.0. A skipped-by-default dogfood test exercises the latter path; it confirms the v0.1.0 stable surface is consumable from a sibling adapter without binding to upstream string literals.

### 8.2 Guardrails AI — complement, not wedge

Guardrails AI (`guardrails-ai/guardrails`) is a mature ($7.5M seed, ~2.9k stars, production-adopted) output-validation framework. Primitives are `Validator` (e.g., `toxic_language`, `competitor_check`, schema coercion), composed into a `Guard` via `.use()`, with `OnFailAction` policies (`REASK` / `FIX` / `FILTER` / `REFRAIN` / `EXCEPTION`) determining behaviour on validation failure. Ships as Python SDK, as a Flask-based server with OpenAI-compatible endpoints, and as a plug-in to LiteLLM proxy and Cloudflare AI Gateway. Validation is best-effort, post-hoc, runtime — there is no certificate or compile-time notion.

**Placement.** Below L1 — same level as OpenMythos in §1. Guardrails validates *what data flows through an agent node* (shape, toxicity, PII, schema); Operon gates validate *how the graph re-shapes under load* (stagnation, contradiction, invariant preservation). The two layers compose cleanly: inside a guarded LangGraph, Guard runs *inside* each agent node (data boundary), Gate runs *between* nodes (process boundary).

**Verdict — complement, not wedge.** Guardrails covers the exact territory §6 marks as explicitly out of scope for Operon ("No LLM output validation for toxicity, schema conformance, or hallucination filtering"). This addendum reaffirms that scope line: Operon should not build output validators, and does not need a Guardrails adapter to claim coverage of that layer — pointing at Guardrails as the sibling framework is the right answer. A joint-demo snippet (one guarded LangGraph node wrapped by both a `Guard` and an Operon pre/post-guard) is the only artefact worth considering, and only if a user asks.

### 8.3 agentflow — L1+L2 wedge candidate, queue position 1

BeraBuilds AgentFlow (`berabuddies/agentflow`, ~1.1k stars, active Apr 2026, no tagged releases) is a Python DAG orchestrator for codex/claude/kimi agents. Primitives are `Graph()` context manager, named agent nodes, operators `>>` (sequence), `fanout` + `merge` (parallel), `on_failure` (loops), `max_iterations` (bounded recursion), and Jinja2 output interpolation. Distinctive secondary feature: a built-in **agent evolution / tuning pipeline** — `agentflow evolve` compiles successful traces into tuned agent versions written to `.agentflow/tuned_agents/`. Ships as Python library + CLI; no plugin surface, no middleware, no config serialization layer (topology is pure Python).

**Placement.** Primary **L1 topology** (`Graph` is a DAG of agent calls — same shape as Swarms/DeerFlow/LangGraph/gascity). Secondary **L2 artefact** — the tuned-agent output is a frozen-prompt-equivalent artefact, landing in the same territory as DSPy compile (§2 Theorem 2) and GEPA optimization (§2 Theorem 3). This is the first L1 framework surveyed with a native L2 evolution loop baked in.

**Verdict — L2 cert hook shipped (2026-05-01); L1 adapter is queue head.** Two separate wedge angles exist:
1. **L1 adapter** — weaker surface than gascity (no hooks; would require wrapping `Graph.run()` or monkey-patching node transitions to inject `StagnationGate`/`IntegrityGate` from `operon-langgraph-gates` v0.1.0). Now queue head for L1 work.
2. **L2 certificate hook for `evolve` — shipped (2026-05-01).** `Certificate.from_agentflow_compile(graph_hash, traces_hash, tuned_agent_hash)` registers a `agentflow_evolve_pinned_inputs` provenance certificate, mirroring the §2 cheap-variant T2 DSPy binding (which shipped the same day). The verifier checks that the three pinned hashes — uncompiled `Graph()`, input traces, and tuned-agent output — are recorded and well-formed; reproducibility itself is checked downstream by re-running `agentflow evolve` and comparing the tuned-agent hash. Code: `operon_ai/convergence/agentflow_certificate.py`. A heavy variant that re-runs `evolve` and checks for hash equality is the natural follow-up but not yet scoped.

If prioritised later, the L2 angle is the distinctive play — no other L1 framework in §1 ships its own optimizer.
