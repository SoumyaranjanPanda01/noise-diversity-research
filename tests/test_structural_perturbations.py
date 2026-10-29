"""Tests for deletion and shuffle perturbations."""

from src.perturbations.structural import apply_deletion, apply_shuffle


def test_reproducibility():
    sample = "The quick brown fox jumps over the lazy dog near the riverbank."
    assert apply_deletion(sample, seed=42) == apply_deletion(sample, seed=42)
    assert apply_shuffle(sample, seed=42) == apply_shuffle(sample, seed=42)


def test_min_words_guardrail():
    short_sample = "Sky is blue."
    # Should not drop words below min_words limit
    assert apply_deletion(short_sample, drop_prob=0.9, min_words=3, seed=42) == short_sample


def test_shuffle_changes_order():
    sample = "Artificial intelligence models learn representations from extensive training datasets."
    shuffled = apply_shuffle(sample, swap_prob=0.8, seed=42)
    assert shuffled != sample
    assert sorted(shuffled.split()) == sorted(sample.split())


def test_empty_string_safety():
    assert apply_deletion("", seed=42) == ""
    assert apply_shuffle("", seed=42) == ""