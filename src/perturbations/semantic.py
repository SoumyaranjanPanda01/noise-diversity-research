"""Semantic perturbations: Contextual Distractor and Synonym Substitution."""

import random
import re
import nltk
from nltk.corpus import wordnet

# Ensure wordnet resources are present
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

NEUTRAL_DISTRACTORS = [
    "Photosynthesis is the process used by plants to convert light into energy.",
    "Mercury is the smallest planet in our solar system.",
    "Paris hosted the Summer Olympic Games in 2024.",
    "The Pacific Ocean is the largest and deepest ocean basin on Earth.",
    "The printing press was developed by Johannes Gutenberg in the fifteenth century.",
    "Mount Everest is the highest mountain peak above sea level.",
    "Jupiter has the shortest day of all the planets in the solar system.",
    "Honey never completely spoils if kept in an airtight container.",
]


def apply_distractor(
    text: str, position: str = "suffix", seed: int | None = None
) -> str:
    """Inserts a fluent, irrelevant distractor sentence into the input."""
    if not text:
        return text

    rng = random.Random(seed)
    distractor = rng.choice(NEUTRAL_DISTRACTORS)

    chosen_pos = position
    if position == "random":
        chosen_pos = rng.choice(["prefix", "suffix"])

    if chosen_pos == "prefix":
        return f"{distractor} {text}"
    return f"{text} {distractor}"


def get_wordnet_synonyms(word: str) -> list[str]:
    """Retrieves single-word lemmas from WordNet."""
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and " " not in name:
                syns.add(name)
    # sorted(): set iteration order depends on PYTHONHASHSEED, which made rng.choice
    # non-reproducible across processes even with a fixed seed.
    return sorted(syns)


def apply_synonym(
    text: str, replace_prob: float = 0.3, seed: int | None = None
) -> str:
    """Replaces eligible non-stop content words with synonyms from WordNet."""
    if not text:
        return text

    rng = random.Random(seed)
    # Basic stopwords to avoid flipping grammatical structure or core interrogatives
    skip_words = {
        "is", "are", "was", "were", "do", "does", "did", "not", "the", "a", "an",
        "of", "on", "in", "to", "for", "with", "at", "by", "from", "up", "about",
        "than", "into", "over", "after", "what", "which", "who", "when", "where", "why", "how"
    }

    tokens = re.findall(r"\w+|[^\w\s]", text)
    transformed = []

    for tok in tokens:
        lower = tok.lower()
        if lower.isalpha() and lower not in skip_words and rng.random() < replace_prob:
            synonyms = get_wordnet_synonyms(lower)
            if synonyms:
                chosen = rng.choice(synonyms)
                if tok.isupper():
                    chosen = chosen.upper()
                elif tok[0].isupper():
                    chosen = chosen.capitalize()
                transformed.append(chosen)
                continue
        transformed.append(tok)

    # Reconstruct string
    result = ""
    for tok in transformed:
        if not result or re.match(r"[^\w\s]", tok):
            result += tok
        else:
            result += " " + tok

    return result