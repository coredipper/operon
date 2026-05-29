"""Tests for the gepa_candidate_improvement certificate theorem.

Certifies that a GEPA-evolved candidate Pareto-dominates its parent on
the validation set: every instance is at-least-as-good and at least one
is strictly better. Used to close GEPA's outer loop with a certificate
(paper-6 follow-up #5).
"""

from operon_ai.core.certificate import (
    Certificate,
    _verify_gepa_candidate_improvement,
    _resolve_verify_fn,
)


def _cert(parent_scores, child_scores):
    return Certificate(
        theorem="gepa_candidate_improvement",
        parameters={"parent_scores": parent_scores, "child_scores": child_scores},
        conclusion="child Pareto-dominates parent on the validation set",
        source="test",
        _verify_fn=_verify_gepa_candidate_improvement,
    )


class TestGepaCandidateImprovement:
    def test_strict_domination_holds(self):
        # Every instance >=, at least one strictly > → dominates.
        result = _cert([0.5, 0.5, 0.5], [0.6, 0.5, 0.7]).verify()
        assert result.holds is True
        assert result.evidence["parent_mean"] == 0.5
        assert round(result.evidence["child_mean"], 4) == 0.6
        assert result.evidence["dominated_count"] == 3   # all instances >=
        assert result.evidence["strict_better_count"] == 2

    def test_uniform_improvement_holds(self):
        result = _cert([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]).verify()
        assert result.holds is True
        assert result.evidence["strict_better_count"] == 3

    def test_exact_tie_does_not_dominate(self):
        # Identical scores: not strictly better anywhere → no domination.
        result = _cert([0.5, 0.6, 0.7], [0.5, 0.6, 0.7]).verify()
        assert result.holds is False
        assert result.evidence["strict_better_count"] == 0
        assert result.evidence["reason"] == "no_strict_improvement"

    def test_one_instance_worse_breaks_domination(self):
        # Better on two, worse on one → not Pareto-domination.
        result = _cert([0.5, 0.5, 0.5], [0.9, 0.9, 0.4]).verify()
        assert result.holds is False
        assert result.evidence["reason"] == "not_dominated"

    def test_empty_scores_fails(self):
        result = _cert([], []).verify()
        assert result.holds is False
        assert result.evidence["reason"] == "empty_evidence"

    def test_length_mismatch_fails(self):
        # Per-instance comparison requires aligned vectors.
        result = _cert([0.5, 0.5], [0.6, 0.7, 0.8]).verify()
        assert result.holds is False
        assert result.evidence["reason"] == "length_mismatch"

    def test_one_sided_empty_parent_is_length_mismatch(self):
        # Scores exist on the child side, so this is unaligned vectors —
        # not "no evidence". Must be length_mismatch, not empty_evidence,
        # and the empty side's mean is reported safely as 0.0.
        result = _cert([], [0.8]).verify()
        assert result.holds is False
        assert result.evidence["reason"] == "length_mismatch"
        assert result.evidence["n_parent"] == 0
        assert result.evidence["n_child"] == 1
        assert result.evidence["parent_mean"] == 0.0
        assert result.evidence["child_mean"] == 0.8

    def test_one_sided_empty_child_is_length_mismatch(self):
        # Symmetric case: parent has scores, child is empty.
        result = _cert([0.5], []).verify()
        assert result.holds is False
        assert result.evidence["reason"] == "length_mismatch"
        assert result.evidence["n_parent"] == 1
        assert result.evidence["n_child"] == 0
        assert result.evidence["parent_mean"] == 0.5
        assert result.evidence["child_mean"] == 0.0

    def test_registered_in_theorem_registry(self):
        # The theorem name must resolve via the built-in registry so the
        # GEPA adapter (and downstream consumers) can look it up by string.
        fn = _resolve_verify_fn("gepa_candidate_improvement")
        assert fn is _verify_gepa_candidate_improvement
