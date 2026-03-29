# Operon TLA+ Specifications

Formal TLA+ specifications for three convergence protocols in the Operon project.
These specs define state machines, safety invariants, and liveness properties that
can be verified with the TLC model checker.

## Specifications

### TemplateExchangeProtocol.tla

Models cross-organism template sharing with trust-weighted adoption (epistemic
vigilance). Organisms export templates, peers adopt them subject to trust and
developmental stage guards, and outcomes feed back into EMA-based trust scores.

Source: `operon_ai/coordination/social_learning.py`

### DevelopmentalGating.tla

Models lifecycle progression through developmental stages (EMBRYONIC, JUVENILE,
ADOLESCENT, MATURE) driven by telomere consumption. Critical periods close
permanently as the organism matures, and tools can only be acquired when the
organism's stage meets the minimum requirement.

Source: `operon_ai/state/development.py`

### ConvergenceDetection.tla

Models intervention-rate convergence detection. Organisms execute stages and may
receive interventions (RETRY, ESCALATE, HALT). When the intervention-to-stage
ratio exceeds a threshold for enough consecutive checks, the organism is halted.

Source: `operon_ai/patterns/watcher.py`

## Installing the TLA+ Toolbox

Download the TLA+ Toolbox from:

https://lamport.azurewebsites.net/tla/toolbox.html

Alternatively, install the VS Code extension `alygin.vscode-tlaplus` for
lightweight editing and model checking.

## Running the TLC Model Checker

1. Open a `.tla` file in the TLA+ Toolbox.
2. Create a new model via **TLC Model Checker > New Model**.
3. Set the constants (see recommended small-model parameters below).
4. Add the safety invariants and liveness properties to check.
5. Click **Run TLC**.

From the command line (requires `tla2tools.jar`):

```
java -jar tla2tools.jar -config SpecName.cfg SpecName.tla
```

## Small-Model Parameters

These parameters keep state spaces tractable for model checking.

### TemplateExchangeProtocol

| Constant       | Value                              |
|----------------|------------------------------------|
| Orgs           | `{o1, o2, o3}`                     |
| Templates      | `{t1, t2, t3}`                     |
| MinStage       | `t1 :> "EMBRYONIC", t2 :> "JUVENILE", t3 :> "ADOLESCENT"` |
| MIN_TRUST      | `0.2`                              |
| DECAY_ALPHA    | `0.3`                              |
| DEFAULT_TRUST  | `0.5`                              |
| MAX_OUTCOMES   | `3`                                |

Invariants to check: `TypeOK`, `TemplateAdoptionSafety`, `TrustMonotonicity`
Properties to check: `QualifyingTemplateEventuallyAdopted`, `TrustConverges` (under `FairSpec`)

### DevelopmentalGating

| Constant           | Value                              |
|--------------------|------------------------------------|
| Orgs               | `{o1, o2}`                         |
| Tools              | `{tool1, tool2, tool3}`            |
| Periods            | `{lang_acq, imprinting}`           |
| ToolMinStage       | `tool1 :> "EMBRYONIC", tool2 :> "JUVENILE", tool3 :> "MATURE"` |
| PeriodClosesAt     | `lang_acq :> "ADOLESCENT", imprinting :> "JUVENILE"` |
| MAX_TELOMERE       | `10`                               |
| JUVENILE_THRESH    | `2`                                |
| ADOLESCENT_THRESH  | `5`                                |
| MATURE_THRESH      | `8`                                |

Invariants to check: `TypeOK`, `CapabilityGating`, `CriticalPeriodIrreversibility`, `StageMonotonicity`, `DevelopmentalProgress`
Properties to check: `EventualMaturity` (under `FairSpec`)

### ConvergenceDetection

| Constant       | Value                              |
|----------------|------------------------------------|
| Orgs           | `{o1, o2, o3}`                     |
| MAX_RATE       | `0.5`                              |
| TOTAL_STAGES   | `5`                                |
| BOUND          | `3`                                |

Invariants to check: `TypeOK`, `BoundedNonConvergence`, `HaltIsTerminal`
Properties to check: `ConvergentOrganismCompletes` (under `FairSpec`)

## Expected Verification Results

With the small-model parameters above, TLC should report:

- **All safety invariants pass** (no counterexamples found).
- **Liveness properties pass** under `FairSpec` (fair scheduling).
- State space for each spec is on the order of thousands to tens of thousands
  of states, completing in seconds to a few minutes.

If a liveness property fails, check that you are using `FairSpec` (which includes
weak fairness) rather than `Spec` as the behavior specification.
