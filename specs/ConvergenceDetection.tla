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

(* StageResult: observe a stage result, increment stages count *)
StageResult(org) ==
    /\ ~halted[org]
    /\ stages[org] < TOTAL_STAGES
    /\ stages' = [stages EXCEPT ![org] = stages[org] + 1]
    /\ UNCHANGED <<interventions, halted>>

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

(* CheckConvergence: evaluate the intervention rate; halt immediately if over threshold.
   Matches watcher.py: immediate halt when intervention_rate > max_rate. *)
CheckConvergence(org) ==
    /\ ~halted[org]
    /\ stages[org] > 0
    /\ InterventionRate(org) > MAX_RATE
    /\ halted' = [halted EXCEPT ![org] = TRUE]
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

(* S1: Once halted, no more stages or interventions occur *)
HaltIsTerminal ==
    [][
        \A org \in Orgs :
            halted[org] = TRUE =>
                /\ stages'[org] = stages[org]
                /\ interventions'[org] = interventions[org]
                /\ halted'[org] = TRUE
    ]_vars

(* S2: If the intervention rate exceeds MAX_RATE and stages > 0 and
        CheckConvergence has been evaluated (i.e. after any step where
        the watcher ran), the organism must be halted.
        Expressed as a temporal property: once rate exceeds threshold,
        the organism is eventually halted. Under WF this is immediate. *)
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
