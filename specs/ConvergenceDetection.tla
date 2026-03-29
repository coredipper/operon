----------------------------- MODULE ConvergenceDetection -----------------------------
(*
  Intervention-rate convergence detection across organisms.

  Models the watcher convergence protocol from operon_ai/patterns/watcher.py.
  Organisms execute stages and may receive interventions (RETRY, ESCALATE, HALT).
  When the ratio of interventions to observed stages exceeds a threshold, the
  organism is halted — signalling non-convergence.

  Reference: Hao et al. (arXiv:2603.15371, BIGMAS) — intervention count as
  convergence proxy.
*)
EXTENDS Naturals, Reals, TLC

CONSTANTS
    Orgs,           \* Set of organism IDs
    MAX_RATE,       \* Maximum tolerable intervention rate (real, e.g. 0.5)
    TOTAL_STAGES,   \* Total stages each organism must complete
    BOUND           \* Max consecutive high-rate steps before forced halt

ASSUME MAX_RATE \in Real /\ MAX_RATE > 0.0 /\ MAX_RATE <= 1.0
ASSUME TOTAL_STAGES \in Nat /\ TOTAL_STAGES > 0
ASSUME BOUND \in Nat /\ BOUND > 0

InterventionKind == {"RETRY", "ESCALATE", "HALT"}

VARIABLES
    interventions,      \* interventions[org]      : Nat, count of interventions
    stages,             \* stages[org]             : Nat, count of stages observed
    halted,             \* halted[org]             : BOOLEAN
    highRateStreak      \* highRateStreak[org]     : Nat, consecutive high-rate checks

vars == <<interventions, stages, halted, highRateStreak>>

-----------------------------------------------------------------------------
(* Helpers *)

InterventionRate(org) ==
    IF stages[org] = 0 THEN 0.0
    ELSE (interventions[org] * 1.0) / (stages[org] * 1.0)

-----------------------------------------------------------------------------
(* Type invariant *)

TypeOK ==
    /\ \A org \in Orgs : interventions[org] \in Nat
    /\ \A org \in Orgs : stages[org] \in 0..TOTAL_STAGES
    /\ \A org \in Orgs : halted[org] \in BOOLEAN
    /\ \A org \in Orgs : highRateStreak[org] \in 0..BOUND

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ interventions  = [org \in Orgs |-> 0]
    /\ stages         = [org \in Orgs |-> 0]
    /\ halted         = [org \in Orgs |-> FALSE]
    /\ highRateStreak = [org \in Orgs |-> 0]

-----------------------------------------------------------------------------
(* Actions *)

(* StageResult: observe a stage result, increment stages count *)
StageResult(org) ==
    /\ ~halted[org]
    /\ stages[org] < TOTAL_STAGES
    /\ stages' = [stages EXCEPT ![org] = stages[org] + 1]
    /\ UNCHANGED <<interventions, halted, highRateStreak>>

(* Intervene: record an intervention of the given kind *)
Intervene(org, kind) ==
    /\ ~halted[org]
    /\ kind \in InterventionKind
    /\ IF kind = "HALT"
       THEN /\ halted' = [halted EXCEPT ![org] = TRUE]
            /\ interventions' = [interventions EXCEPT ![org] = interventions[org] + 1]
            /\ UNCHANGED <<stages, highRateStreak>>
       ELSE /\ interventions' = [interventions EXCEPT ![org] = interventions[org] + 1]
            /\ UNCHANGED <<stages, halted, highRateStreak>>

(* CheckConvergence: evaluate the intervention rate; halt if over threshold *)
CheckConvergence(org) ==
    /\ ~halted[org]
    /\ stages[org] > 0
    /\ IF InterventionRate(org) > MAX_RATE
       THEN /\ highRateStreak' = [highRateStreak EXCEPT
                ![org] = highRateStreak[org] + 1]
            /\ IF highRateStreak[org] + 1 >= BOUND
               THEN halted' = [halted EXCEPT ![org] = TRUE]
               ELSE UNCHANGED halted
       ELSE /\ highRateStreak' = [highRateStreak EXCEPT ![org] = 0]
            /\ UNCHANGED halted
    /\ UNCHANGED <<interventions, stages>>

-----------------------------------------------------------------------------
(* Next-state relation *)

Next ==
    \/ \E org \in Orgs : StageResult(org)
    \/ \E org \in Orgs : \E kind \in InterventionKind : Intervene(org, kind)
    \/ \E org \in Orgs : CheckConvergence(org)

Spec == Init /\ [][Next]_vars

FairSpec == Spec
    /\ WF_vars(\E org \in Orgs : StageResult(org))
    /\ WF_vars(\E org \in Orgs : CheckConvergence(org))

-----------------------------------------------------------------------------
(* Safety invariants *)

(* S1: If the intervention rate exceeds MAX_RATE for BOUND consecutive checks,
        the organism must be halted. *)
BoundedNonConvergence ==
    \A org \in Orgs :
        highRateStreak[org] >= BOUND => halted[org] = TRUE

(* S2: Once halted, no more stages or interventions occur *)
HaltIsTerminal ==
    [][
        \A org \in Orgs :
            halted[org] = TRUE =>
                /\ stages'[org] = stages[org]
                /\ interventions'[org] = interventions[org]
                /\ halted'[org] = TRUE
    ]_vars

-----------------------------------------------------------------------------
(* Liveness (requires FairSpec) *)

(* L1: If an organism's intervention rate stays below the threshold,
        it eventually completes all stages. *)
ConvergentOrganismCompletes ==
    \A org \in Orgs :
        ([]~halted[org]) ~> (stages[org] = TOTAL_STAGES)

=============================================================================
