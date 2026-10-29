"""Semantic and contextual perturbations: Distractor and Synonym Substitution."""

import random
import re

# Verified neutral, label-independent distractor sentences
DISTRACTOR_POOL = [
    "The 2024 Summer Olympics were hosted in Paris.",
    "Mount Everest is the highest mountain above sea level.",
    "Honey never actually spoils if stored properly.",
    "The Pacific Ocean is the largest and deepest ocean basin.",
    "Photosynthesis is the process used by plants to convert light into energy.",
    "The human heart pumps roughly 2,000 gallons of blood each day.",
    "Mercury is the smallest planet in our solar system.",
    "Paper was originally invented during the Han dynasty in China.",
]

# Lightweight synonym dictionary for common descriptive/relational words
SYNONYM_MAP = {
    "large": "big",
    "big": "large",
    "small": "tiny",
    "tiny": "small",
    "quick": "rapid",
    "fast": "rapid",
    "rapid": "fast",
    "accurate": "correct",
    "correct": "accurate",
    "begin": "start",
    "start": "begin",
    "create": "produce",
    "produce": "create",
    "build": "construct",
    "construct": "build",
    "difficult": "challenging",
    "hard": "difficult",
    "important": "significant",
    "primary": "main",
    "main": "primary",
}


def apply_distractor(text: str, seed: int | None = None) -> str:
    """Inserts a fluent, irrelevant sentence into the text context.
    
    Position is randomized across beginning, middle, or end.
    """
    if not text:
        return text

    rng = random.Random(seed)
    distractor = rng.choice(DISTRACTOR_POOL)

    # Split into sentences or rough chunks
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
        return f"{distractor} {text}"

    insert_idx = rng.randint(0, len(sentences))
    sentences.insert(insert_idx, distractor)

    return " ".join(sentences)


def apply_synonym(text: str, replace_prob: float = 0.3, seed: int | None = None) -> str:
    """Replaces words with synonyms using a curated label-preserving dictionary."""
    if not text:
        return text

    rng = random.Random(seed)
    words = text.split()
    perturbed_words = []

    for word in words:
        clean_word = word.strip(".,!?;:\"'").lower()
        if clean_word in SYNONYM_MAP and rng.random() < replace_prob:
            replacement = SYNONYM_MAP[clean_word]
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            if word and word[-1] in ".,!?;:":
                replacement = replacement + word[-1]
            perturbed_words.append(replacement)
        else:
            perturbed_words.append(word)

    return " ".join(perturbed_words)