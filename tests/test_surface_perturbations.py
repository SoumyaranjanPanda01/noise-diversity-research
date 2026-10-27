"""Tests for typo and formatting perturbations."""

from src.perturbations.surface import apply_formatting, apply_typo


def test_reproducibility():
    sample = "Is it true that the earth orbits the sun?"
    # Same seed should give identical outputs
    assert apply_typo(sample, seed=42) == apply_typo(sample, seed=42)
    assert apply_formatting(sample, seed=42) == apply_formatting(sample, seed=42)


def test_seed_variation():
    sample = "Is it true that the earth orbits the sun?"
    # Different seeds should usually generate distinct variations
    out1 = apply_typo(sample, char_prob=0.15, seed=1)
    out2 = apply_typo(sample, char_prob=0.15, seed=2)
    assert out1 != out2


def test_empty_string_safety():
    assert apply_typo("", seed=42) == ""
    assert apply_formatting("", seed=42) == ""