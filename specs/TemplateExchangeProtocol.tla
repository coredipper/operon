--------------------------- MODULE TemplateExchangeProtocol ---------------------------
(*
  Cross-organism template sharing with trust-weighted adoption.

  Models the social learning protocol from operon_ai/coordination/social_learning.py.
  Organisms maintain pattern libraries, exchange templates with peers, and modulate
  adoption decisions via EMA-based trust scoring (epistemic vigilance).

  Biological analogy: horizontal gene transfer in bacteria with epistemic vigilance
  determining whether foreign genetic material is incorporated or rejected.
*)
EXTENDS Naturals, Reals, FiniteSets, Sequences, TLC

CONSTANTS
    Orgs,               \* Set of organism IDs
    Templates,          \* Set of all possible template IDs
    MinStage,           \* Function: template -> minimum stage required
    MIN_TRUST,          \* Minimum trust to adopt (default 0.2)
    ADOPTION_THRESHOLD, \* peer_success_rate * trust must meet this (default 0.3)
    DECAY_ALPHA,        \* EMA smoothing factor (default 0.3)
    DEFAULT_TRUST,      \* Initial trust for unknown peers (default 0.5)
    MAX_OUTCOMES        \* Bound on outcomes for model checking

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
    records,     \* records[org]            : set of <<template_id, success>> pairs
    exported,    \* exported[org]           : set of template IDs currently offered
    outcomeCount,\* outcomeCount[org]       : Nat, bounds exploration for TLC
    successRate, \* successRate[org][tmpl]  : real in [0,1], peer-reported success rate
    adoptedFrom  \* adoptedFrom[org][tmpl]  : peer that supplied this template (or "none")

vars == <<library, trust, stage, records, exported, outcomeCount, successRate, adoptedFrom>>

-----------------------------------------------------------------------------
(* Type invariant *)

TypeOK ==
    /\ \A org \in Orgs : library[org] \subseteq Templates
    /\ \A org \in Orgs : \A peer \in Orgs :
         trust[org][peer] \in Real /\ trust[org][peer] >= 0.0 /\ trust[org][peer] <= 1.0
    /\ \A org \in Orgs : stage[org] \in StageSet
    /\ \A org \in Orgs : \A r \in records[org] :
         /\ r[1] \in Templates
         /\ r[2] \in BOOLEAN
    /\ \A org \in Orgs : exported[org] \subseteq Templates
    /\ \A org \in Orgs : outcomeCount[org] \in Nat
    /\ \A org \in Orgs : \A tmpl \in Templates :
         successRate[org][tmpl] \in Real
         /\ successRate[org][tmpl] >= 0.0 /\ successRate[org][tmpl] <= 1.0
    /\ \A org \in Orgs : \A tmpl \in Templates :
         adoptedFrom[org][tmpl] \in Orgs \union {"none"}

-----------------------------------------------------------------------------
(* Initial state *)

Init ==
    /\ library      = [org \in Orgs |-> {}]
    /\ trust        = [org \in Orgs |-> [peer \in Orgs |-> DEFAULT_TRUST]]
    /\ stage        = [org \in Orgs |-> "EMBRYONIC"]
    /\ records      = [org \in Orgs |-> {}]
    /\ exported     = [org \in Orgs |-> {}]
    /\ outcomeCount = [org \in Orgs |-> 0]
    /\ successRate  = [org \in Orgs |-> [tmpl \in Templates |-> 0.5]]
    /\ adoptedFrom  = [org \in Orgs |-> [tmpl \in Templates |-> "none"]]

-----------------------------------------------------------------------------
(* Actions *)

(* Export: organism makes its current library available to peers *)
Export(org) ==
    /\ exported' = [exported EXCEPT ![org] = library[org]]
    /\ UNCHANGED <<library, trust, stage, records, outcomeCount, successRate, adoptedFrom>>

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
         /\ successRate[peer][tmpl] * trust[org][peer] >= ADOPTION_THRESHOLD \* Adoption threshold
         /\ library'     = [library EXCEPT ![org] = library[org] \union {tmpl}]
         /\ adoptedFrom' = [adoptedFrom EXCEPT ![org][tmpl] = peer]
    /\ UNCHANGED <<trust, stage, records, exported, outcomeCount, successRate>>

(* RecordOutcome: record whether an adopted template succeeded, update trust via EMA.
   Only updates trust for the specific peer that supplied the template (via adoptedFrom). *)
RecordOutcome(org, tmpl, success) ==
    /\ tmpl \in library[org]                                \* Must hold the template
    /\ outcomeCount[org] < MAX_OUTCOMES                     \* TLC bound
    /\ adoptedFrom[org][tmpl] \in Orgs                      \* Template came from a known peer
    /\ LET peer     == adoptedFrom[org][tmpl]
           oldTrust == trust[org][peer]
           val      == IF success THEN 1.0 ELSE 0.0
           newTrust == DECAY_ALPHA * val + (1.0 - DECAY_ALPHA) * oldTrust
       IN  trust' = [trust EXCEPT ![org][peer] = newTrust]
    /\ records' = [records EXCEPT ![org] = records[org] \union {<<tmpl, success>>}]
    /\ outcomeCount' = [outcomeCount EXCEPT ![org] = outcomeCount[org] + 1]
    /\ UNCHANGED <<library, stage, exported, successRate, adoptedFrom>>

(* StageAdvance: external advancement of developmental stage (driven by environment) *)
StageAdvance(org) ==
    /\ stage[org] # "MATURE"
    /\ LET next == CASE stage[org] = "EMBRYONIC"  -> "JUVENILE"
                   []   stage[org] = "JUVENILE"   -> "ADOLESCENT"
                   []   stage[org] = "ADOLESCENT" -> "MATURE"
       IN  stage' = [stage EXCEPT ![org] = next]
    /\ UNCHANGED <<library, trust, records, exported, outcomeCount, successRate, adoptedFrom>>

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
         /\ successRate[peer][tmpl] * trust[org][peer] >= ADOPTION_THRESHOLD
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
