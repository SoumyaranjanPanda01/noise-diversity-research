"""Tests for distractor and synonym perturbations."""

from src.perturbations.semantic import (
    NEUTRAL_DISTRACTORS,
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
    assert any(distractor in perturbed for distractor in NEUTRAL_DISTRACTORS)
    assert len(perturbed) > len(sample)


def test_synonym_replacement():
    sample = "The primary goal is to build a large model."
    perturbed = apply_synonym(sample, replace_prob=1.0, seed=42)
    # Specific WordNet picks are an implementation detail; check the property instead.
    assert perturbed != sample
    assert len(perturbed.split()) == len(sample.split())
    assert perturbed.split()[3] == "is"  # stopwords untouched


def test_empty_string_safety():
    assert apply_distractor("", seed=42) == ""
    assert apply_synonym("", seed=42) == ""

def test_synonym_deterministic_across_hash_seeds():
    """Regression: synonym choice must not depend on PYTHONHASHSEED (set ordering)."""
    import os
    import subprocess
    import sys

    code = ("from src.perturbations.semantic import apply_synonym;"
            "print(apply_synonym('is the primary goal to build a large quick model', replace_prob=1.0, seed=7))")
    outs = set()
    for hs in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": "."}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                text=True, check=True).stdout)
    assert len(outs) == 1
