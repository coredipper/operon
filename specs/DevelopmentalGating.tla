------------------------------ MODULE DevelopmentalGating ------------------------------
(*
  Lifecycle progression with critical periods and capability gating.

  Models the developmental staging protocol from operon_ai/state/development.py.
  Organisms consume telomere capacity over time, advancing through developmental
  stages. Critical periods open and close permanently. Tools can only be acquired
  when the organism's stage meets the tool's minimum requirement.

  Biological analogy: neurodevelopmental critical periods where specific neural
  circuits are maximally plastic (language acquisition, imprinting).
*)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS
    Orgs,               \* Set of organism IDs
    Tools,              \* Set of tool IDs
    Periods,            \* Set of critical period names
    ToolMinStage,       \* Function: tool -> minimum stage required
    PeriodOpensAt,      \* Function: period -> stage at which it opens
    PeriodClosesAt,     \* Function: period -> stage at which it closes
    MAX_TELOMERE,       \* Initial telomere capacity (natural number)
    JUVENILE_THRESH,    \* Telomere units consumed to reach JUVENILE
    ADOLESCENT_THRESH,  \* Telomere units consumed to reach ADOLESCENT
    MATURE_THRESH       \* Telomere units consumed to reach MATURE

ASSUME MAX_TELOMERE \in Nat /\ MAX_TELOMERE > 0
ASSUME JUVENILE_THRESH \in Nat /\ JUVENILE_THRESH > 0
ASSUME ADOLESCENT_THRESH \in Nat /\ ADOLESCENT_THRESH > JUVENILE_THRESH
ASSUME MATURE_THRESH \in Nat /\ MATURE_THRESH > ADOLESCENT_THRESH
ASSUME MATURE_THRESH <= MAX_TELOMERE

StageSet == {"EMBRYONIC", "JUVENILE", "ADOLESCENT", "MATURE"}

StageOrd(s) ==
    CASE s = "EMBRYONIC"  -> 0
    []   s = "JUVENILE"   -> 1
    []   s = "ADOLESCENT" -> 2
    []   s = "MATURE"     -> 3

StageGEQ(a, b) == StageOrd(a) >= StageOrd(b)

VARIABLES
    telomere,   \* telomere[org]  : Nat, remaining capacity
    stage,      \* stage[org]     : element of StageSet
    periods,    \* periods[org]   : function period -> {"pending", "open", "closed"}
    tools       \* tools[org]     : set of acquired tool IDs

vars == <<telomere, stage, periods, tools>>

-----------------------------------------------------------------------------
(* Derived: how many units have been consumed *)
Consumed(org) == MAX_TELOMERE - telomere[org]

(* Compute the stage from consumed count *)
StageFor(consumed) ==
    IF consumed >= MATURE_THRESH     THEN "MATURE"
    ELSE IF consumed >= ADOLESCENT_THRESH THEN "ADOLESCENT"
    ELSE IF consumed >= JUVENILE_THRESH   THEN "JUVENILE"
    ELSE "EMBRYONIC"

(* Whether a period should be open at a given stage *)
PeriodShouldBeOpen(period, s) == StageGEQ(s, PeriodOpensAt[period])

(* Whether a period should be closed at a given stage *)
PeriodShouldBeClosed(period, s) == StageGEQ(s, PeriodClosesAt[period])

-----------------------------------------------------------------------------
(* Type invariant *)

TypeOK ==
    /\ \A org \in Orgs : telomere[org] \in 0..MAX_TELOMERE
    /\ \A org \in Orgs : stage[org] \in StageSet
    /\ \A org \in Orgs : \A p \in Periods : periods[org][p] \in {"pending", "open", "closed"}
    /\ \A org \in Orgs : tools[org] \subseteq Tools

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ telomere = [org \in Orgs |-> MAX_TELOMERE]
    /\ stage    = [org \in Orgs |-> "EMBRYONIC"]
    /\ periods  = [org \in Orgs |-> [p \in Periods |->
                    IF PeriodShouldBeClosed(p, "EMBRYONIC") THEN "closed"
                    ELSE IF PeriodShouldBeOpen(p, "EMBRYONIC") THEN "open"
                    ELSE "pending"]]
    /\ tools    = [org \in Orgs |-> {}]

-----------------------------------------------------------------------------
(* Actions *)

(* Tick: consume one telomere unit, advance stage if threshold crossed,
   close any critical periods whose closes_at stage has been reached. *)
Tick(org) ==
    /\ telomere[org] > 0
    /\ telomere' = [telomere EXCEPT ![org] = telomere[org] - 1]
    /\ LET newConsumed == Consumed(org) + 1
           newStage    == StageFor(newConsumed)
       IN  /\ stage' = [stage EXCEPT ![org] = newStage]
           /\ periods' = [periods EXCEPT ![org] =
                [p \in Periods |->
                    IF PeriodShouldBeClosed(p, newStage)
                    THEN "closed"
                    ELSE IF periods[org][p] = "pending" /\ PeriodShouldBeOpen(p, newStage)
                    THEN "open"
                    ELSE periods[org][p]
                ]]
    /\ UNCHANGED tools

(* AcquireTool: acquire a tool if the organism's stage meets the minimum *)
AcquireTool(org, tool) ==
    /\ tool \notin tools[org]
    /\ StageGEQ(stage[org], ToolMinStage[tool])
    /\ tools' = [tools EXCEPT ![org] = tools[org] \union {tool}]
    /\ UNCHANGED <<telomere, stage, periods>>

(* OpenPeriod: transition a pending critical period to open when stage >= opensAt *)
OpenPeriod(org, period) ==
    /\ periods[org][period] = "pending"
    /\ PeriodShouldBeOpen(period, stage[org])
    /\ periods' = [periods EXCEPT ![org][period] = "open"]
    /\ UNCHANGED <<telomere, stage, tools>>

(* ClosePeriod: explicitly close a critical period (idempotent) *)
ClosePeriod(org, period) ==
    /\ periods[org][period] = "open"
    /\ PeriodShouldBeClosed(period, stage[org])
    /\ periods' = [periods EXCEPT ![org][period] = "closed"]
    /\ UNCHANGED <<telomere, stage, tools>>

(* Scaffold: teacher shares a template with learner, filtered by stage.
   Modelled as teacher enabling learner to acquire a tool. *)
Scaffold(teacher, learner) ==
    /\ teacher # learner
    /\ \E tool \in Tools :
        /\ tool \notin tools[learner]
        /\ tool \in tools[teacher]                          \* Teacher has it
        /\ StageGEQ(stage[learner], ToolMinStage[tool])     \* Learner meets min_stage
        /\ tools' = [tools EXCEPT ![learner] = tools[learner] \union {tool}]
    /\ UNCHANGED <<telomere, stage, periods>>

-----------------------------------------------------------------------------
(* Next-state relation *)

Next ==
    \/ \E org \in Orgs : Tick(org)
    \/ \E org \in Orgs : \E tool \in Tools : AcquireTool(org, tool)
    \/ \E org \in Orgs : \E p \in Periods : OpenPeriod(org, p)
    \/ \E org \in Orgs : \E p \in Periods : ClosePeriod(org, p)
    \/ \E t \in Orgs : \E l \in Orgs : Scaffold(t, l)

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ WF_vars(\E org \in Orgs : Tick(org))

-----------------------------------------------------------------------------
(* Safety invariants *)

(* S1: A tool's min_stage never exceeds the holder's current stage *)
CapabilityGating ==
    \A org \in Orgs : \A tool \in tools[org] :
        StageGEQ(stage[org], ToolMinStage[tool])

(* S2: Once a critical period is closed, it never reopens.
   Also, periods only advance: pending -> open -> closed, never backwards. *)
CriticalPeriodIrreversibility ==
    [][
        \A org \in Orgs : \A p \in Periods :
            /\ (periods[org][p] = "closed" => periods'[org][p] = "closed")
            /\ (periods[org][p] = "open"   => periods'[org][p] \in {"open", "closed"})
    ]_vars

(* S3: Stages only advance, never regress *)
StageMonotonicity ==
    [][
        \A org \in Orgs :
            StageOrd(stage'[org]) >= StageOrd(stage[org])
    ]_vars

(* S4: Telomere decreases on every Tick *)
DevelopmentalProgress ==
    [][
        \A org \in Orgs :
            telomere'[org] <= telomere[org]
    ]_vars

-----------------------------------------------------------------------------
(* Liveness (requires FairSpec) *)

(* L1: An organism that keeps ticking eventually reaches MATURE *)
EventualMaturity ==
    \A org \in Orgs :
        <>(stage[org] = "MATURE")

=============================================================================
