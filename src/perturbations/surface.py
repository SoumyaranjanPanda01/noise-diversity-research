"""Surface-level perturbations: Typos and Formatting."""

import random
import re
import string


def apply_typo(text: str, char_prob: float = 0.05, seed: int | None = None) -> str:
    """Applies character-level typos (swap, drop, keyboard neighbor).
    
    Preserves basic word boundaries and token lengths.
    """
    if not text:
        return text

    rng = random.Random(seed)
    chars = list(text)
    n = len(chars)

    # Keyboard layout adjacency map for realistic typos
    adjacent_keys = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
        'k': 'ijlm', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }

    i = 0
    while i < n:
        # Skip spaces and punctuation for swap/substitute
        if chars[i].isalpha() and rng.random() < char_prob:
            op = rng.choice(["swap", "replace", "delete"])
            
            if op == "swap" and i + 1 < n and chars[i + 1].isalpha():
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 1  # Skip next char to avoid double swap
            elif op == "replace":
                lower_c = chars[i].lower()
                if lower_c in adjacent_keys:
                    sub = rng.choice(adjacent_keys[lower_c])
                    chars[i] = sub.upper() if chars[i].isupper() else sub
            elif op == "delete" and len(chars) > 3:
                chars.pop(i)
                n -= 1
                continue
        i += 1

    return "".join(chars)


def apply_formatting(text: str, seed: int | None = None) -> str:
    """Applies formatting perturbations: irregular casing, random whitespaces, and punctuation changes."""
    if not text:
        return text

    rng = random.Random(seed)
    words = text.split()
    perturbed_words = []

    for word in words:
        rand_val = rng.random()
        # 1. Random casing perturbation
        if rand_val < 0.2:
            word = word.upper()
        elif rand_val < 0.4:
            word = word.lower()
        elif rand_val < 0.5:
            # Alternating casing: e.g. "tHaT"
            word = "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(word))

        # 2. Random punctuation alteration
        if word and word[-1] in string.punctuation and rng.random() < 0.3:
            punc_choice = rng.choice(["remove", "duplicate", "space_before"])
            if punc_choice == "remove":
                word = word[:-1]
            elif punc_choice == "duplicate":
                word = word + word[-1]
            elif punc_choice == "space_before":
                word = word[:-1] + " " + word[-1]

        perturbed_words.append(word)

    # 3. Random irregular whitespace insertion
    result = ""
    for idx, w in enumerate(perturbed_words):
        if idx == 0:
            result = w
        else:
            space = "  " if rng.random() < 0.15 else " "
            result += space + w

    return result