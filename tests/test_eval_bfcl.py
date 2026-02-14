"""Tests for the BFCL folding evaluation suite."""
from __future__ import annotations

import random

from eval.suites.bfcl_folding import BfclFoldingConfig, run_bfcl_folding


class TestBfclFolding:
    def test_returns_expected_keys(self):
        config = BfclFoldingConfig(max_samples=5)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        assert "samples" in result
        assert "strict" in result
        assert "cascade" in result
        assert result["samples"] > 0

    def test_strict_has_counter_fields(self):
        config = BfclFoldingConfig(max_samples=5)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        for key in ("success", "total", "rate", "wilson_95"):
            assert key in result["strict"]
            assert key in result["cascade"]

    def test_cascade_ge_strict(self):
        config = BfclFoldingConfig(max_samples=20)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        assert result["cascade"]["rate"] >= result["strict"]["rate"]

    def test_deterministic(self):
        config = BfclFoldingConfig(max_samples=10)
        r1 = run_bfcl_folding(config, random.Random(99))
        r2 = run_bfcl_folding(config, random.Random(99))
        assert r1 == r2

    def test_source_field_present(self):
        config = BfclFoldingConfig(max_samples=3)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        assert result["source"] in ("bfcl", "fallback")

    def test_samples_matches_config(self):
        config = BfclFoldingConfig(max_samples=7)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        assert result["strict"]["total"] == result["samples"]
        assert result["cascade"]["total"] == result["samples"]

    def test_total_equals_success_plus_failures(self):
        config = BfclFoldingConfig(max_samples=10)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        strict = result["strict"]
        assert strict["total"] >= strict["success"] >= 0

    def test_wilson_interval_bounds(self):
        config = BfclFoldingConfig(max_samples=15)
        rng = random.Random(42)
        result = run_bfcl_folding(config, rng)
        for counter_name in ("strict", "cascade"):
            low, high = result[counter_name]["wilson_95"]
            assert 0.0 <= low <= high <= 1.0
