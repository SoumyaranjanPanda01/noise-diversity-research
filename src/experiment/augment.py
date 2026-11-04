"""Generic edit-based augmentation baseline (EDA, Wei & Zou 2019), made holdout-safe.

Standard EDA mixes synonym replacement (SR), random insertion (RI), random swap (RS)
and random deletion (RD). SR, RS and RD coincide with our synonym, shuffle and
deletion families, so using all four would leak held-out families into training.
Each fold therefore only uses the EDA operations that do not correspond to its
unseen families (deviation D2 in docs/deviations.md):

* fold_a (unseen: shuffle, distractor, synonym) -> RD, RI
* fold_b (unseen: typo, formatting, deletion)   -> RS, RI, SR
"""

import random

from src.perturbations.semantic import get_wordnet_synonyms

ALPHA = 0.1  # fraction of words edited per operation (EDA default)

FOLD_OPS = {
    "fold_a": ("rd", "ri"),
    "fold_b": ("rs", "ri", "sr"),
}

_SKIP = {"is", "are", "was", "were", "do", "does", "did", "can", "the", "a", "an", "not"}


def _n_edits(words: list[str]) -> int:
    return max(1, round(ALPHA * len(words)))


def random_deletion(words: list[str], rng: random.Random) -> list[str]:
    if len(words) <= 3:
        return words
    kept = [w for w in words if rng.random() >= ALPHA]
    return kept if len(kept) >= 3 else words


def random_insertion(words: list[str], rng: random.Random) -> list[str]:
    """Inserts copies of words already in the sentence (no new content -> no distractor leak)."""
    out = list(words)
    candidates = [w for w in words if w.lower() not in _SKIP] or words
    for _ in range(_n_edits(words)):
        out.insert(rng.randrange(len(out) + 1), rng.choice(candidates))
    return out


def random_swap(words: list[str], rng: random.Random) -> list[str]:
    out = list(words)
    if len(out) < 2:
        return out
    for _ in range(_n_edits(words)):
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out


def synonym_replacement(words: list[str], rng: random.Random) -> list[str]:
    out = list(words)
    idxs = [i for i, w in enumerate(words) if w.isalpha() and w.lower() not in _SKIP]
    rng.shuffle(idxs)
    done = 0
    for i in idxs:
        syns = get_wordnet_synonyms(words[i].lower())
        if syns:
            out[i] = rng.choice(sorted(syns))
            done += 1
        if done >= _n_edits(words):
            break
    return out


OPS = {"rd": random_deletion, "ri": random_insertion, "rs": random_swap, "sr": synonym_replacement}


def eda_augment(text: str, fold: str, seed: int) -> tuple[str, str]:
    """Returns (augmented_text, op_name) using one fold-permitted EDA operation."""
    rng = random.Random(seed)
    op = rng.choice(FOLD_OPS[fold])
    words = text.split()
    if not words:
        return text, op
    return " ".join(OPS[op](words, rng)), op
