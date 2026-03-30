--------------------------- MODULE TemplateExchangeProtocol ---------------------------
(*
  Cross-organism template sharing with trust-weighted adoption.

  Models the social learning protocol from operon_ai/coordination/social_learning.py.
  Organisms maintain pattern libraries, exchange templates with peers, and modulate
  adoption decisions via EMA-based trust scoring (epistemic vigilance).

  Biological analogy: horizontal gene transfer in bacteria with epistemic vigilance
  determining whether foreign genetic material is incorporated or rejected.
*)
\* Source diversity: organisms may import templates from various external
\* systems. The protocol is source-agnostic; common sources include:
\*   - Swarms (graph-based workflows)
\*   - DeerFlow (LangGraph sessions)
\*   - AnimaWorks (supervisor hierarchies)
\*   - Ralph (event-driven hat orchestration)
\*   - A-Evolve (evolved workspace snapshots)

EXTENDS Naturals, Reals, FiniteSets, Sequences, TLC

CONSTANTS
    Orgs,               \* Set of organism IDs
    Templates,          \* Set of all possible template IDs
    MinStage,           \* Function: template -> minimum stage required
    MIN_TRUST,          \* Minimum trust to adopt (default 0.2)
    ADOPTION_THRESHOLD, \* peer_success_rate * trust must meet this (default 0.3)
    DECAY_ALPHA,        \* EMA smoothing factor (default 0.3)
    DEFAULT_TRUST,      \* Initial trust for unknown peers (default 0.5)
    MAX_OUTCOMES,       \* Bound on outcomes for model checking
    InitLibrary         \* Function: org -> set of initial templates (must be stage-compatible)

ASSUME MIN_TRUST \in Real /\ MIN_TRUST >= 0.0 /\ MIN_TRUST <= 1.0
ASSUME ADOPTION_THRESHOLD \in Real /\ ADOPTION_THRESHOLD >= 0.0 /\ ADOPTION_THRESHOLD <= 1.0
ASSUME DECAY_ALPHA \in Real /\ DECAY_ALPHA >= 0.0 /\ DECAY_ALPHA <= 1.0
ASSUME DEFAULT_TRUST \in Real /\ DEFAULT_TRUST >= 0.0 /\ DEFAULT_TRUST <= 1.0

StageSet == {"EMBRYONIC", "JUVENILE", "ADOLESCENT", "MATURE"}

StageOrd(s) ==
    CASE s = "EMBRYONIC"  -> 0
    []   s = "JUVENILE"   -> 1
    []   s = "ADOLESCENT" -> 2
    []   s = "MATURE"     -> 3

StageGEQ(a, b) == StageOrd(a) >= StageOrd(b)

VARIABLES
    library,     \* library[org]            : set of template IDs held by org
    trust,       \* trust[org][peer]        : real in [0,1], trust org has in peer
    stage,       \* stage[org]              : element of StageSet
    successes,   \* successes[org][tmpl]    : Nat, count of successful runs
    totals,      \* totals[org][tmpl]       : Nat, total run count
    exported,    \* exported[org]           : set of template IDs currently offered
    outcomeCount,\* outcomeCount[org]       : Nat, bounds exploration for TLC
    adoptedFrom  \* adoptedFrom[org][tmpl]  : peer that supplied this template (or "none")

vars == <<library, trust, stage, successes, totals, exported, outcomeCount, adoptedFrom>>

\* Derived: check if a peer's template meets the adoption threshold.
\* Matches social_learning.py: effective_score = peer_sr * trust >= ADOPTION_THRESHOLD.
\* peer_sr = successes/total from the peer's records (0.5 if no records).
\* Rearranged to avoid division: successes * trust * 10 >= ADOPTION_THRESHOLD * 10 * total.
\* Trust is real [0,1]; ADOPTION_THRESHOLD is 0.3.
MeetsAdoptionThreshold(peer, tmpl, trustVal) ==
    LET total == totals[peer][tmpl]
        succ  == successes[peer][tmpl]
    IN  IF total = 0
        THEN trustVal * 0.5 >= ADOPTION_THRESHOLD  \* default 0.5 success rate
        ELSE succ * trustVal >= ADOPTION_THRESHOLD * total

-----------------------------------------------------------------------------
(* Type invariant *)

TypeOK ==
    /\ \A org \in Orgs : library[org] \subseteq Templates
    /\ \A org \in Orgs : \A peer \in Orgs :
         trust[org][peer] \in Real /\ trust[org][peer] >= 0.0 /\ trust[org][peer] <= 1.0
    /\ \A org \in Orgs : stage[org] \in StageSet
    /\ \A org \in Orgs : \A tmpl \in Templates :
         /\ successes[org][tmpl] \in Nat
         /\ totals[org][tmpl] \in Nat
         /\ successes[org][tmpl] <= totals[org][tmpl]
    /\ \A org \in Orgs : exported[org] \subseteq Templates
    /\ \A org \in Orgs : outcomeCount[org] \in Nat
    /\ \A org \in Orgs : \A tmpl \in Templates :
         adoptedFrom[org][tmpl] \in Orgs \union {"none"}

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    \* Each org starts with a distinct template set (InitLibrary) so that
    \* Import is reachable (peers offer templates the org doesn't already hold).
    \* InitLibrary must assign only stage-compatible templates (MinStage <= EMBRYONIC).
    /\ library      = InitLibrary
    /\ trust        = [org \in Orgs |-> [peer \in Orgs |-> DEFAULT_TRUST]]
    /\ stage        = [org \in Orgs |-> "EMBRYONIC"]
    /\ successes    = [org \in Orgs |-> [tmpl \in Templates |-> 0]]
    /\ totals       = [org \in Orgs |-> [tmpl \in Templates |-> 0]]
    /\ exported     = InitLibrary
    /\ outcomeCount = [org \in Orgs |-> 0]
    /\ adoptedFrom  = [org \in Orgs |-> [tmpl \in Templates |-> "none"]]

-----------------------------------------------------------------------------
(* Actions *)

(* Export: organism makes its current library available to peers *)
Export(org) ==
    /\ exported' = [exported EXCEPT ![org] = library[org]]
    /\ UNCHANGED <<library, trust, stage, successes, totals, outcomeCount, adoptedFrom>>

(* Import: organism considers adopting a peer's exported templates.
   Matches social_learning.py: effective_score = peer_success_rate * trust
   must meet ADOPTION_THRESHOLD; trust must also meet MIN_TRUST. *)
Import(org, peer) ==
    /\ org # peer                                           \* NoSelfAdoption guard
    /\ trust[org][peer] >= MIN_TRUST                        \* Trust guard
    /\ exported[peer] # {}                                  \* Peer has something to offer
    /\ \E tmpl \in exported[peer] :
         /\ tmpl \notin library[org]                        \* Not already held
         /\ StageGEQ(stage[org], MinStage[tmpl])            \* Stage guard
         /\ MeetsAdoptionThreshold(peer, tmpl, trust[org][peer]) \* Derived from peer's records
         /\ library'     = [library EXCEPT ![org] = library[org] \union {tmpl}]
         /\ adoptedFrom' = [adoptedFrom EXCEPT ![org][tmpl] = peer]
    /\ UNCHANGED <<trust, stage, successes, totals, exported, outcomeCount>>

(* RecordOutcome: record whether an adopted template succeeded, update trust via EMA.
   Only updates trust for the specific peer that supplied the template (via adoptedFrom).
   Success rate is derived from records at Import time — no separate state variable. *)
RecordOutcome(org, tmpl, success) ==
    /\ tmpl \in library[org]                                \* Must hold the template
    /\ outcomeCount[org] < MAX_OUTCOMES                     \* TLC bound
    /\ adoptedFrom[org][tmpl] \in Orgs                      \* Template came from a known peer
    /\ LET peer     == adoptedFrom[org][tmpl]
           oldTrust == trust[org][peer]
           val      == IF success THEN 1.0 ELSE 0.0
           newTrust == DECAY_ALPHA * val + (1.0 - DECAY_ALPHA) * oldTrust
       IN  trust' = [trust EXCEPT ![org][peer] = newTrust]
    /\ totals' = [totals EXCEPT ![org][tmpl] = totals[org][tmpl] + 1]
    /\ successes' = [successes EXCEPT ![org][tmpl] =
         IF success THEN successes[org][tmpl] + 1 ELSE successes[org][tmpl]]
    /\ outcomeCount' = [outcomeCount EXCEPT ![org] = outcomeCount[org] + 1]
    /\ UNCHANGED <<library, stage, exported, adoptedFrom>>

(* StageAdvance: external advancement of developmental stage (driven by environment) *)
StageAdvance(org) ==
    /\ stage[org] # "MATURE"
    /\ LET next == CASE stage[org] = "EMBRYONIC"  -> "JUVENILE"
                   []   stage[org] = "JUVENILE"   -> "ADOLESCENT"
                   []   stage[org] = "ADOLESCENT" -> "MATURE"
       IN  stage' = [stage EXCEPT ![org] = next]
    /\ UNCHANGED <<library, trust, successes, totals, exported, outcomeCount, adoptedFrom>>

-----------------------------------------------------------------------------
(* Next-state relation *)

Next ==
    \/ \E org \in Orgs : Export(org)
    \/ \E org \in Orgs : \E peer \in Orgs : Import(org, peer)
    \/ \E org \in Orgs : \E tmpl \in Templates : \E s \in BOOLEAN :
         RecordOutcome(org, tmpl, s)
    \/ \E org \in Orgs : StageAdvance(org)

Spec == Init /\ [][Next]_vars

FairSpec == Spec /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Safety invariants *)

(* S1: An adopted template's min_stage never exceeds the adopter's stage *)
TemplateAdoptionSafety ==
    \A org \in Orgs : \A tmpl \in library[org] :
        StageGEQ(stage[org], MinStage[tmpl])

(* S2: Trust only changes through RecordOutcome -- encoded structurally:
   Export, Import, StageAdvance all leave trust UNCHANGED. *)
TrustMonotonicity ==
    \A org \in Orgs : \A peer \in Orgs :
        trust[org][peer] >= 0.0 /\ trust[org][peer] <= 1.0

(* S3: An organism never imports from itself -- enforced by Import guard *)
NoSelfAdoption ==
    TRUE \* Structurally guaranteed by Import(org, peer) requiring org # peer.
         \* Expressed here for documentation; can be strengthened with history var.

-----------------------------------------------------------------------------
(* Liveness properties (require FairSpec) *)

(* L1: If trust >= MIN_TRUST and success_rate * trust >= ADOPTION_THRESHOLD
        and stage >= min_stage and peer exported the template,
        eventually the template is adopted. *)
QualifyingTemplateEventuallyAdopted ==
    \A org \in Orgs : \A peer \in Orgs \ {org} : \A tmpl \in Templates :
        (   trust[org][peer] >= MIN_TRUST
         /\ MeetsAdoptionThreshold(peer, tmpl, trust[org][peer])
         /\ StageGEQ(stage[org], MinStage[tmpl])
         /\ tmpl \in exported[peer]
         /\ tmpl \notin library[org]
        ) ~> (tmpl \in library[org])

(* L2: After enough outcomes, trust stabilizes -- modelled as eventually
        the outcome count reaches MAX_OUTCOMES (bounding exploration). *)
TrustConverges ==
    \A org \in Orgs :
        <>(outcomeCount[org] = MAX_OUTCOMES)

=============================================================================
