------------------------------ MODULE EvolutionGating ------------------------------
(*
  Evolution loop with monotonic score gating.

  Models the A-Evolve Solve->Observe->Evolve->Gate->Reload loop. Organisms
  maintain a workspace version and a benchmark score. Mutations are generated
  nondeterministically; a gate action accepts the mutation only when the new
  score meets or exceeds the current score. Rejected mutations are rolled back.

  Biological analogy: positive selection in directed evolution — only
  beneficial mutations survive the fitness gate.

  See: https://github.com/A-EVO-Lab/a-evolve
*)
EXTENDS Naturals, FiniteSets, Reals, TLC

CONSTANTS
    Orgs,               \* Set of organism IDs
    MAX_VERSIONS,       \* Upper bound on workspace version number
    MIN_IMPROVEMENT,    \* Minimum score improvement to accept (>= 0, can be 0)
    ScoreSet            \* Finite set of candidate scores for model checking

ASSUME MAX_VERSIONS \in Nat /\ MAX_VERSIONS > 0
ASSUME MIN_IMPROVEMENT \in Real /\ MIN_IMPROVEMENT >= 0.0

VARIABLES
    workspace,    \* workspace[org]    : Nat, current version number
    score,        \* score[org]        : Real, current benchmark score
    pending,      \* pending[org]      : Real | "none", score of pending mutation
    accepted      \* accepted[org]     : Nat, count of accepted mutations

vars == <<workspace, score, pending, accepted>>

-----------------------------------------------------------------------------
(* Type invariant *)

TypeOK ==
    /\ \A org \in Orgs : workspace[org] \in 0..MAX_VERSIONS
    /\ \A org \in Orgs : score[org] \in Real /\ score[org] >= 0.0
    /\ \A org \in Orgs : pending[org] \in Real \union {"none"}
    /\ \A org \in Orgs : accepted[org] \in Nat

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ workspace = [org \in Orgs |-> 0]
    /\ score     = [org \in Orgs |-> 0.0]
    /\ pending   = [org \in Orgs |-> "none"]
    /\ accepted  = [org \in Orgs |-> 0]

-----------------------------------------------------------------------------
(* Actions *)

(* Evolve: generate a pending mutation with a nondeterministic score.
   Models the Solve+Observe phase of A-Evolve — the organism produces a
   candidate workspace snapshot and measures its benchmark score. *)
Evolve(org) ==
    /\ pending[org] = "none"                          \* No mutation in flight
    /\ workspace[org] < MAX_VERSIONS                  \* Room for another version
    /\ \E s \in ScoreSet :
         /\ s >= 0.0                                  \* Non-negative score
         /\ pending' = [pending EXCEPT ![org] = s]
    /\ UNCHANGED <<workspace, score, accepted>>

(* Gate: accept the pending mutation if its score meets or exceeds the
   current score (plus optional MIN_IMPROVEMENT). Increments the workspace
   version and updates the organism's score. *)
Gate(org) ==
    /\ pending[org] # "none"
    /\ pending[org] >= score[org] + MIN_IMPROVEMENT   \* Fitness gate
    /\ workspace' = [workspace EXCEPT ![org] = workspace[org] + 1]
    /\ score'     = [score EXCEPT ![org] = pending[org]]
    /\ accepted'  = [accepted EXCEPT ![org] = accepted[org] + 1]
    /\ pending'   = [pending EXCEPT ![org] = "none"]

(* Rollback: discard the pending mutation without changing workspace or score.
   Models the case where the mutation does not improve fitness. *)
Rollback(org) ==
    /\ pending[org] # "none"
    /\ pending' = [pending EXCEPT ![org] = "none"]
    /\ UNCHANGED <<workspace, score, accepted>>

-----------------------------------------------------------------------------
(* Next-state relation *)

Next ==
    \/ \E org \in Orgs : Evolve(org)
    \/ \E org \in Orgs : Gate(org)
    \/ \E org \in Orgs : Rollback(org)

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Safety properties *)
(* TypeOK and VersionBound are state invariants (checked under Invariants in TLC).
   MonotonicScore and GateBeforeDeploy are temporal safety properties
   (checked under Properties in TLC). *)

(* S1: Score never decreases after an accepted mutation.
   Encoded as a temporal property: in every step, if workspace advances,
   the new score is at least as high as the old score. *)
MonotonicScore ==
    [][
        \A org \in Orgs :
            workspace'[org] > workspace[org] => score'[org] >= score[org]
    ]_vars

(* S2: Workspace only increments through Gate — never jumps or decrements.
   Structurally guaranteed because only Gate modifies workspace, and it
   increments by exactly 1. Expressed as a temporal property for TLC. *)
GateBeforeDeploy ==
    [][
        \A org \in Orgs :
            workspace'[org] # workspace[org] =>
                workspace'[org] = workspace[org] + 1
    ]_vars

(* S3: Workspace version never exceeds MAX_VERSIONS. *)
VersionBound ==
    \A org \in Orgs : workspace[org] <= MAX_VERSIONS

-----------------------------------------------------------------------------
(* Liveness discussion *)

(* This spec intentionally provides NO liveness property. The nondeterministic
   score generation in Evolve and the choice between Gate/Rollback mean that
   weak fairness on Next does not guarantee eventual acceptance:
     - Evolve may never generate an improving score.
     - Rollback may always discard qualifying mutations.
   Evolution liveness ("organism eventually improves") requires external
   assumptions about the score generator and rollback policy that are outside
   the scope of this structural model. The safety properties (MonotonicScore,
   GateBeforeDeploy) and invariants (TypeOK, VersionBound) are the
   verifiable guarantees. *)

=============================================================================
