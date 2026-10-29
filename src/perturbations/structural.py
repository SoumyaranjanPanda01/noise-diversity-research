"""Structural and lexical perturbations: Deletion and Shuffle."""

import random


def apply_deletion(
    text: str, drop_prob: float = 0.15, min_words: int = 3, seed: int | None = None
) -> str:
    """Randomly drops words at rate drop_prob, preserving at least min_words.
    
    Guards against empty outputs or extreme length collapse.
    """
    if not text:
        return text

    rng = random.Random(seed)
    words = text.split()

    # If the text is already short, do not delete to protect label/meaning
    if len(words) <= min_words:
        return text

    surviving_words = []
    for word in words:
        if rng.random() >= drop_prob:
            surviving_words.append(word)

    # Fallback guardrail: if too many were dropped, randomly restore from original
    if len(surviving_words) < min_words:
        surviving_words = rng.sample(words, min_words)

    return " ".join(surviving_words)


def apply_shuffle(text: str, swap_prob: float = 0.2, seed: int | None = None) -> str:
    """Performs local adjacent word swaps to alter syntax without destroying meaning."""
    if not text:
        return text

    rng = random.Random(seed)
    words = text.split()
    n = len(words)

    if n < 2:
        return text

    i = 0
    while i < n - 1:
        if rng.random() < swap_prob:
            words[i], words[i + 1] = words[i + 1], words[i]
            i += 1  # Skip next word to prevent chain swapping across the entire sentence
        i += 1

    return " ".join(words)