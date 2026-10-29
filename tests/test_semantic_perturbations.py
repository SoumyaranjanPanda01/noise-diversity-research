"""Tests for distractor and synonym perturbations."""

from src.perturbations.semantic import (
    DISTRACTOR_POOL,
    apply_distractor,
    apply_synonym,
)


def test_reproducibility():
    sample = "The quick study produced accurate results on difficult tasks."
    assert apply_distractor(sample, seed=42) == apply_distractor(sample, seed=42)
    assert apply_synonym(sample, seed=42) == apply_synonym(sample, seed=42)


def test_distractor_inserted():
    sample = "Climate patterns vary across regions. Temperature affects crop yield."
    perturbed = apply_distractor(sample, seed=123)
    # Ensure one of the pool sentences is now present
    assert any(distractor in perturbed for distractor in DISTRACTOR_POOL)
    assert len(perturbed) > len(sample)


def test_synonym_replacement():
    sample = "The primary goal is to build a large model."
    perturbed = apply_synonym(sample, replace_prob=1.0, seed=42)
    assert "main" in perturbed or "construct" in perturbed or "big" in perturbed


def test_empty_string_safety():
    assert apply_distractor("", seed=42) == ""
    assert apply_synonym("", seed=42) == ""