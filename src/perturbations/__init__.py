"""Unified perturbation interface."""

from src.perturbations.semantic import apply_distractor, apply_synonym
from src.perturbations.structural import apply_deletion, apply_shuffle
from src.perturbations.surface import apply_formatting, apply_typo

PERTURBATION_FUNCTIONS = {
    "typo": apply_typo,
    "formatting": apply_formatting,
    "deletion": apply_deletion,
    "shuffle": apply_shuffle,
    "distractor": apply_distractor,
    "synonym": apply_synonym,
}


def perturb_text(text: str, family: str, seed: int | None = None) -> str:
    """Applies a named perturbation family with a fixed random seed."""
    if family not in PERTURBATION_FUNCTIONS:
        raise ValueError(f"Unknown family: {family}. Available: {list(PERTURBATION_FUNCTIONS.keys())}")
    return PERTURBATION_FUNCTIONS[family](text, seed=seed)