# Operon TLA+ Specifications

Formal TLA+ specifications for four convergence protocols in the Operon project.
These specs define state machines, safety invariants, and liveness properties that
can be verified with the TLC model checker. Each spec ships with a `.cfg` file
containing small-model parameters ready for TLC.

## Specifications

### TemplateExchangeProtocol.tla

Models cross-organism template sharing with trust-weighted adoption (epistemic
vigilance). Organisms export templates, peers adopt them subject to trust,
peer-reported success rate, and developmental stage guards, and outcomes feed
back into EMA-based trust scores. The `adoptedFrom` state tracks which peer
supplied each template so that `RecordOutcome` updates the correct peer's trust.

Source: `operon_ai/coordination/social_learning.py`

**Checked properties:**

| Name | Kind | Description |
|------|------|-------------|
| `TypeOK` | INVARIANT | Type correctness of all state variables |
| `TemplateAdoptionSafety` | INVARIANT | Adopted template's min\_stage never exceeds adopter's stage |
| `TrustMonotonicity` | INVARIANT | Trust values remain in [0, 1] |
| `QualifyingTemplateEventuallyAdopted` | PROPERTY (liveness) | Qualifying templates are eventually adopted (requires DEFAULT\_TRUST >= 0.6 to be satisfiable; commented out in default .cfg) |
| `TrustConverges` | PROPERTY (liveness) | Outcome counts eventually reach MAX\_OUTCOMES (same precondition; commented out in default .cfg) |

### DevelopmentalGating.tla

Models lifecycle progression through developmental stages (EMBRYONIC, JUVENILE,
ADOLESCENT, MATURE) driven by telomere consumption. Critical periods start
"pending", become "open" when the organism reaches the period's `opensAt` stage,
and close permanently at the `closesAt` stage. Tools can only be acquired when
the organism's stage meets the minimum requirement.

Source: `operon_ai/state/development.py`

**Checked properties:**

| Name | Kind | Description |
|------|------|-------------|
| `TypeOK` | INVARIANT | Type correctness of all state variables |
| `CapabilityGating` | INVARIANT | Tool's min\_stage never exceeds holder's stage |
| `CriticalPeriodIrreversibility` | PROPERTY (temporal safety) | Closed periods never reopen; transitions only advance |
| `StageMonotonicity` | PROPERTY (temporal safety) | Stages only advance, never regress |
| `DevelopmentalProgress` | PROPERTY (temporal safety) | Telomere never increases |
| `EventualMaturity` | PROPERTY (liveness) | Every organism eventually reaches MATURE |

### ConvergenceDetection.tla

Models intervention-rate convergence detection. Organisms execute stages and may
receive interventions (RETRY, ESCALATE, HALT). When the intervention-to-stage
ratio exceeds the threshold, the organism is halted immediately.

Source: `operon_ai/patterns/watcher.py`

**Checked properties:**

| Name | Kind | Description |
|------|------|-------------|
| `TypeOK` | INVARIANT | Type correctness of all state variables |
| `HaltIsTerminal` | PROPERTY (temporal safety) | Once halted, no further state changes occur |
| `BoundedNonConvergence` | PROPERTY (temporal safety) | Rate exceeding MAX\_RATE leads to halt |
| `ConvergentOrganismCompletes` | PROPERTY (liveness) | Non-halted organisms eventually complete all stages |

### EvolutionGating.tla

Models the A-Evolve Solve->Observe->Evolve->Gate->Reload loop. Organisms
maintain a workspace version and a benchmark score. Mutations are generated
nondeterministically; a gate action accepts the mutation only when the new
score meets or exceeds the current score. Rejected mutations are rolled back.

Source: [A-Evolve](https://github.com/A-EVO-Lab/a-evolve)

**Checked properties:**

| Name | Kind | Description |
|------|------|-------------|
| `TypeOK` | INVARIANT | Type correctness of all state variables |
| `VersionBound` | INVARIANT | Workspace version never exceeds MAX\_VERSIONS |
| `MonotonicScore` | PROPERTY (temporal safety) | Score never decreases after an accepted mutation |
| `GateBeforeDeploy` | PROPERTY (temporal safety) | Workspace only increments by exactly 1 through Gate |

> **No liveness property.** Evolution liveness ("organism eventually improves")
> requires external assumptions about the score generator and rollback policy
> that are outside the model's scope. See the comment in the spec for details.

## Installing the TLA+ Toolbox

Download the TLA+ Toolbox from:

https://lamport.azurewebsites.net/tla/toolbox.html

Alternatively, install the VS Code extension `alygin.vscode-tlaplus` for
lightweight editing and model checking.

To run TLC from the command line, download `tla2tools.jar` from:

https://github.com/tlaplus/tlaplus/releases

## Reproducing Verification

Each spec has a matching `.cfg` file with small-model parameters. To verify
a spec from the command line:

### Step 1: Download tla2tools.jar

```bash
# Download the latest release (one-time setup)
curl -L -o tla2tools.jar \
  https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

### Step 2: Run TLC on each spec

From the `specs/` directory:

```bash
# TemplateExchangeProtocol (safety only — default parameters)
java -jar tla2tools.jar -config TemplateExchangeProtocol.cfg TemplateExchangeProtocol.tla

# TemplateExchangeProtocol (safety + liveness — elevated trust)
java -jar tla2tools.jar -config TemplateExchangeProtocol-liveness.cfg TemplateExchangeProtocol.tla

# DevelopmentalGating
java -jar tla2tools.jar -config DevelopmentalGating.cfg DevelopmentalGating.tla

# ConvergenceDetection
java -jar tla2tools.jar -config ConvergenceDetection.cfg ConvergenceDetection.tla

# EvolutionGating (see note below)
java -jar tla2tools.jar -config EvolutionGating.cfg EvolutionGating.tla
```

### Step 3: Interpret results

TLC should report no errors for all safety invariants and temporal properties.
For specs with liveness properties (TemplateExchangeProtocol, DevelopmentalGating,
ConvergenceDetection), ensure the `SPECIFICATION` line in the `.cfg` file is set
to `FairSpec` (which includes weak fairness). Without fairness, liveness checks
may fail spuriously.

### Note on EvolutionGating scores

The `Evolve` action quantifies over a finite `ScoreSet` constant (default
`{0, 1, 2, 3}` in the `.cfg` file). Adjust the set size to trade off
state-space coverage against model-checking time. Use integer scores to
avoid TLC's limitations with real-number enumeration.

## Small-Model Parameters

These parameters keep state spaces tractable for model checking. They match
the values in the `.cfg` files.

### TemplateExchangeProtocol

| Constant           | Value                              |
|--------------------|------------------------------------|
| Orgs               | `{o1, o2}`                         |
| Templates          | `{t1, t2}`                         |
| MinStage           | `[t1 \|-> "EMBRYONIC", t2 \|-> "EMBRYONIC"]` |
| MIN_TRUST          | `0.2`                              |
| ADOPTION_THRESHOLD | `0.3`                              |
| DECAY_ALPHA        | `0.3`                              |
| DEFAULT_TRUST      | `0.5`                              |
| MAX_OUTCOMES       | `3`                                |
| InitLibrary        | `[o1 \|-> {t1}, o2 \|-> {t2}]`    |

### DevelopmentalGating

| Constant           | Value                              |
|--------------------|------------------------------------|
| Orgs               | `{o1, o2}`                         |
| Tools              | `{tool1, tool2}`                   |
| Periods            | `{lang_acq, imprinting}`           |
| ToolMinStage       | `[tool1 \|-> "EMBRYONIC", tool2 \|-> "JUVENILE"]` |
| PeriodOpensAt      | `[lang_acq \|-> "EMBRYONIC", imprinting \|-> "EMBRYONIC"]` |
| PeriodClosesAt     | `[lang_acq \|-> "ADOLESCENT", imprinting \|-> "JUVENILE"]` |
| MAX_TELOMERE       | `10`                               |
| JUVENILE_THRESH    | `2`                                |
| ADOLESCENT_THRESH  | `5`                                |
| MATURE_THRESH      | `8`                                |

### ConvergenceDetection

| Constant       | Value                              |
|----------------|------------------------------------|
| Orgs           | `{o1, o2}`                         |
| MAX_RATE       | `0.5`                              |
| TOTAL_STAGES   | `5`                                |

### EvolutionGating

| Constant           | Value                              |
|--------------------|------------------------------------|
| Orgs               | `{o1, o2}`                         |
| MAX_VERSIONS       | `5`                                |
| MIN_IMPROVEMENT    | `0`                                |
| ScoreSet           | `{0, 1, 2, 3}`                    |

## Property Classification Guide

TLC distinguishes between two kinds of checked formulas:

- **INVARIANT**: A state predicate evaluated at every reachable state. Examples:
  `TypeOK`, `CapabilityGating`, `TemplateAdoptionSafety`, `TrustMonotonicity`,
  `VersionBound`.

- **PROPERTY**: A temporal formula evaluated over complete behaviors. This
  includes both temporal safety formulas (using `[][...]_vars`) and liveness
  formulas (using `<>` or `~>`). Examples: `MonotonicScore`, `HaltIsTerminal`,
  `EventualMaturity`, `QualifyingTemplateEventuallyAdopted`.

Placing a temporal formula under INVARIANT (or a state predicate under PROPERTY)
will produce incorrect results or TLC errors.

## Expected Verification Results

With the small-model parameters above, TLC should report:

- **All safety invariants pass** (no counterexamples found).
- **Liveness properties pass** under `FairSpec` (fair scheduling) for specs
  that define them (TemplateExchangeProtocol, DevelopmentalGating,
  ConvergenceDetection). EvolutionGating defines no liveness property --
  it checks safety invariants (`TypeOK`, `VersionBound`) and temporal
  safety properties (`MonotonicScore`, `GateBeforeDeploy`) only.
- State space for each spec is on the order of thousands to tens of thousands
  of states, completing in seconds to a few minutes.

If a liveness property fails, check that you are using `FairSpec` (which includes
weak fairness) rather than `Spec` as the behavior specification.
