"""Tests for Thymus baseline training."""
import pytest
from datetime import datetime
from operon_ai.surveillance.types import MHCPeptide
from operon_ai.surveillance.thymus import (
    Thymus, BaselineProfile, SelectionResult,
)


def make_peptide(
    agent_id: str = "test",
    output_length_mean: float = 100.0,
    output_length_std: float = 10.0,
    response_time_mean: float = 0.5,
    response_time_std: float = 0.1,
    vocabulary_hash: str = "abc123",
    structure_hash: str = "def456",
    confidence_mean: float = 0.9,
    confidence_std: float = 0.05,
    error_rate: float = 0.01,
) -> MHCPeptide:
    """Helper to create test peptides."""
    return MHCPeptide(
        agent_id=agent_id,
        timestamp=datetime.utcnow(),
        output_length_mean=output_length_mean,
        output_length_std=output_length_std,
        response_time_mean=response_time_mean,
        response_time_std=response_time_std,
        vocabulary_hash=vocabulary_hash,
        structure_hash=structure_hash,
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        error_rate=error_rate,
        error_types=(),
    )


class TestSelectionResult:
    def test_selection_results_exist(self):
        assert SelectionResult.POSITIVE.value == "positive"
        assert SelectionResult.NEGATIVE.value == "negative"
        assert SelectionResult.ANERGIC.value == "anergic"
        assert SelectionResult.INSUFFICIENT_DATA.value == "insufficient"


class TestBaselineProfile:
    def test_create_profile(self):
        profile = BaselineProfile(
            agent_id="test",
            output_length_bounds=(80.0, 120.0),
            response_time_bounds=(0.3, 0.7),
            confidence_bounds=(0.8, 1.0),
            error_rate_max=0.1,
            valid_vocabulary_hashes={"abc", "def"},
            valid_structure_hashes={"json", "plain"},
            canary_accuracy_min=0.9,
        )
        assert profile.agent_id == "test"
        assert profile.output_length_bounds == (80.0, 120.0)

    def test_check_bounds_pass(self):
        profile = BaselineProfile(
            agent_id="test",
            output_length_bounds=(80.0, 120.0),
            response_time_bounds=(0.3, 0.7),
            confidence_bounds=(0.8, 1.0),
            error_rate_max=0.1,
            valid_vocabulary_hashes={"abc123"},
            valid_structure_hashes={"def456"},
            canary_accuracy_min=0.8,
        )
        peptide = make_peptide()  # Within all bounds
        violations = profile.check(peptide)
        assert len(violations) == 0

    def test_check_bounds_output_length_violation(self):
        profile = BaselineProfile(
            agent_id="test",
            output_length_bounds=(80.0, 90.0),  # Peptide has 100.0
            response_time_bounds=(0.3, 0.7),
            confidence_bounds=(0.8, 1.0),
            error_rate_max=0.1,
            valid_vocabulary_hashes={"abc123"},
            valid_structure_hashes={"def456"},
            canary_accuracy_min=0.8,
        )
        peptide = make_peptide()
        violations = profile.check(peptide)
        assert len(violations) == 1
        assert "output_length" in violations[0]

    def test_check_bounds_vocabulary_violation(self):
        profile = BaselineProfile(
            agent_id="test",
            output_length_bounds=(80.0, 120.0),
            response_time_bounds=(0.3, 0.7),
            confidence_bounds=(0.8, 1.0),
            error_rate_max=0.1,
            valid_vocabulary_hashes={"xyz789"},  # Different hash
            valid_structure_hashes={"def456"},
            canary_accuracy_min=0.8,
        )
        peptide = make_peptide()
        violations = profile.check(peptide)
        assert len(violations) == 1
        assert "vocabulary_hash" in violations[0]

    def test_check_bounds_multiple_violations(self):
        profile = BaselineProfile(
            agent_id="test",
            output_length_bounds=(200.0, 300.0),  # Way off
            response_time_bounds=(1.0, 2.0),       # Way off
            confidence_bounds=(0.99, 1.0),         # Too strict
            error_rate_max=0.001,                  # Too strict
            valid_vocabulary_hashes={"xyz"},
            valid_structure_hashes={"xyz"},
            canary_accuracy_min=0.99,
        )
        peptide = make_peptide()
        violations = profile.check(peptide)
        assert len(violations) >= 4


class TestThymus:
    def test_create_thymus(self):
        thymus = Thymus(min_training_samples=5, tolerance=2.0)
        assert thymus.min_training_samples == 5
        assert thymus.tolerance == 2.0

    def test_train_insufficient_samples(self):
        thymus = Thymus(min_training_samples=5)
        samples = [make_peptide() for _ in range(3)]  # Only 3 samples
        profile, result = thymus.train("test_agent", samples)
        assert result == SelectionResult.INSUFFICIENT_DATA
        assert profile is None

    def test_train_success(self):
        thymus = Thymus(min_training_samples=3, tolerance=2.0)
        samples = [
            make_peptide(output_length_mean=100.0, output_length_std=10.0),
            make_peptide(output_length_mean=105.0, output_length_std=10.0),
            make_peptide(output_length_mean=95.0, output_length_std=10.0),
        ]
        profile, result = thymus.train("test_agent", samples)
        assert result == SelectionResult.POSITIVE
        assert profile is not None
        assert profile.agent_id == "test_agent"

    def test_train_calculates_bounds(self):
        thymus = Thymus(min_training_samples=3, tolerance=2.0)
        samples = [
            make_peptide(output_length_mean=100.0, output_length_std=5.0),
            make_peptide(output_length_mean=100.0, output_length_std=5.0),
            make_peptide(output_length_mean=100.0, output_length_std=5.0),
        ]
        profile, _ = thymus.train("test_agent", samples)

        # Bounds should be mean ± tolerance × std
        # mean=100, std=5, tolerance=2 -> 100 ± 10 = (90, 110)
        low, high = profile.output_length_bounds
        assert low == pytest.approx(90.0, rel=0.1)
        assert high == pytest.approx(110.0, rel=0.1)

    def test_train_collects_hashes(self):
        thymus = Thymus(min_training_samples=2)
        samples = [
            make_peptide(vocabulary_hash="hash1", structure_hash="struct1"),
            make_peptide(vocabulary_hash="hash2", structure_hash="struct2"),
        ]
        profile, _ = thymus.train("test_agent", samples)
        assert "hash1" in profile.valid_vocabulary_hashes
        assert "hash2" in profile.valid_vocabulary_hashes
        assert "struct1" in profile.valid_structure_hashes
        assert "struct2" in profile.valid_structure_hashes

    def test_train_high_variance_anergic(self):
        thymus = Thymus(min_training_samples=3, variance_threshold=0.5)
        # Very high variance in outputs - anergic (useless for detection)
        samples = [
            make_peptide(output_length_mean=10.0, output_length_std=50.0),
            make_peptide(output_length_mean=500.0, output_length_std=200.0),
            make_peptide(output_length_mean=1000.0, output_length_std=100.0),
        ]
        profile, result = thymus.train("test_agent", samples)
        assert result == SelectionResult.ANERGIC

    def test_train_stores_canary_minimum(self):
        thymus = Thymus(min_training_samples=3)
        samples = [
            make_peptide(),
            make_peptide(),
            make_peptide(),
        ]
        # Add canary_accuracy to samples (need to recreate with canary)
        samples_with_canary = []
        for s in samples:
            p = MHCPeptide(
                agent_id=s.agent_id,
                timestamp=s.timestamp,
                output_length_mean=s.output_length_mean,
                output_length_std=s.output_length_std,
                response_time_mean=s.response_time_mean,
                response_time_std=s.response_time_std,
                vocabulary_hash=s.vocabulary_hash,
                structure_hash=s.structure_hash,
                confidence_mean=s.confidence_mean,
                confidence_std=s.confidence_std,
                error_rate=s.error_rate,
                error_types=s.error_types,
                canary_accuracy=0.95,
            )
            samples_with_canary.append(p)

        profile, _ = thymus.train("test_agent", samples_with_canary)
        assert profile.canary_accuracy_min == pytest.approx(0.95 * 0.9, rel=0.1)  # 90% of observed
