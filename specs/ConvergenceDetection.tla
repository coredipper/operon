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
    TOTAL_STAGES    \* Total stages each organism must complete

ASSUME MAX_RATE \in Real /\ MAX_RATE > 0.0 /\ MAX_RATE <= 1.0
ASSUME TOTAL_STAGES \in Nat /\ TOTAL_STAGES > 0

InterventionKind == {"RETRY", "ESCALATE", "HALT"}

VARIABLES
    interventions,      \* interventions[org]      : Nat, count of interventions
    stages,             \* stages[org]             : Nat, count of stages observed
    halted              \* halted[org]             : BOOLEAN

vars == <<interventions, stages, halted>>

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

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ interventions  = [org \in Orgs |-> 0]
    /\ stages         = [org \in Orgs |-> 0]
    /\ halted         = [org \in Orgs |-> FALSE]

-----------------------------------------------------------------------------
(* Actions *)

(* StageResult: observe a stage result, increment stages count, then check convergence.
   Matches watcher.py: on_stage_result increments total_stages and immediately checks
   the intervention rate in the same atomic step.  If the rate exceeds MAX_RATE the
   organism is halted and the HALT intervention is recorded (Fix 2 + Fix 3). *)
StageResult(org) ==
    /\ ~halted[org]
    /\ stages[org] < TOTAL_STAGES
    /\ LET newStages == stages[org] + 1
           rate      == IF newStages = 0 THEN 0.0
                        ELSE (interventions[org] * 1.0) / (newStages * 1.0)
       IN  /\ stages' = [stages EXCEPT ![org] = newStages]
           /\ IF rate > MAX_RATE
              THEN /\ halted'        = [halted EXCEPT ![org] = TRUE]
                   /\ interventions' = [interventions EXCEPT ![org] = interventions[org] + 1]
              ELSE UNCHANGED <<halted, interventions>>

(* Intervene: record an intervention of the given kind *)
Intervene(org, kind) ==
    /\ ~halted[org]
    /\ kind \in InterventionKind
    /\ IF kind = "HALT"
       THEN /\ halted' = [halted EXCEPT ![org] = TRUE]
            /\ interventions' = [interventions EXCEPT ![org] = interventions[org] + 1]
            /\ UNCHANGED stages
       ELSE /\ interventions' = [interventions EXCEPT ![org] = interventions[org] + 1]
            /\ UNCHANGED <<stages, halted>>

-----------------------------------------------------------------------------
(* Next-state relation *)

Next ==
    \/ \E org \in Orgs : StageResult(org)
    \/ \E org \in Orgs : \E kind \in InterventionKind : Intervene(org, kind)

Spec == Init /\ [][Next]_vars

FairSpec == Spec
    /\ WF_vars(\E org \in Orgs : StageResult(org))

-----------------------------------------------------------------------------
(* Safety invariants *)

(* S1: Once halted, no more stages or interventions occur *)
HaltIsTerminal ==
    [][
        \A org \in Orgs :
            halted[org] = TRUE =>
                /\ stages'[org] = stages[org]
                /\ interventions'[org] = interventions[org]
                /\ halted'[org] = TRUE
    ]_vars

(* S2: If the intervention rate exceeds MAX_RATE at the moment a stage
        completes, the organism is halted in the same atomic step.
        Expressed as a safety invariant: any non-halted organism with
        stages > 0 whose rate exceeds MAX_RATE cannot persist — the
        next StageResult that pushed the rate over will have already set
        halted.  We keep the liveness form for backward compatibility. *)
BoundedNonConvergence ==
    \A org \in Orgs :
        (stages[org] > 0 /\ InterventionRate(org) > MAX_RATE /\ ~halted[org])
            ~> halted[org] = TRUE

-----------------------------------------------------------------------------
(* Liveness (requires FairSpec) *)

(* L1: If an organism's intervention rate stays below the threshold,
        it eventually completes all stages. *)
ConvergentOrganismCompletes ==
    \A org \in Orgs :
        ([]~halted[org]) ~> (stages[org] = TOTAL_STAGES)

=============================================================================
