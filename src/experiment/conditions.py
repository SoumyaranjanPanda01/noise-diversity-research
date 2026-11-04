"""Registry of the training conditions in protocol v1 (see docs/protocol_v1.md §5).

Every condition trains on exactly 4,000 examples drawn from the same frozen
train split, so step counts are matched across conditions.
"""

from dataclasses import dataclass, field
from pathlib import Path

PERTURBED_DIR = Path("data/perturbed")
SPLITS_DIR = Path("data/splits")

FOLDS = {
    "fold_a": {"seen": ["typo", "formatting", "deletion"], "unseen": ["shuffle", "distractor", "synonym"]},
    "fold_b": {"seen": ["shuffle", "distractor", "synonym"], "unseen": ["typo", "formatting", "deletion"]},
}
ALL_FAMILIES = FOLDS["fold_a"]["seen"] + FOLDS["fold_b"]["seen"]
SWEEP_FRACTIONS = [0.1, 0.25, 1.0]


@dataclass(frozen=True)
class Condition:
    name: str              # unique id, e.g. "fold_a/mixed3_f0.50"
    category: str          # clean | single | mixed3 | sweep | neftune | augment
    fold: str | None       # None => shared across folds
    data_path: Path
    corruption_fraction: float | None
    neftune: bool = False
    families: tuple[str, ...] = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


def sweep_path(fold: str, frac: float) -> Path:
    return PERTURBED_DIR / f"train_{fold}_mixed3_f{frac:.2f}.jsonl"


def augment_path(fold: str) -> Path:
    return PERTURBED_DIR / f"train_{fold}_augment.jsonl"


def build_conditions() -> list[Condition]:
    conds = [
        Condition("shared/clean", "clean", None, SPLITS_DIR / "train.jsonl", 0.0),
        # NEFTune on clean data is fold-independent, so it is shared like the clean
        # baseline (deviation D1 in docs/deviations.md).
        Condition("shared/neftune", "neftune", None, SPLITS_DIR / "train.jsonl", 0.0, neftune=True),
    ]
    for fold, spec in FOLDS.items():
        for fam in spec["seen"]:
            conds.append(Condition(f"{fold}/single_{fam}", "single", fold,
                                   PERTURBED_DIR / f"train_{fold}_{fam}.jsonl", 0.5, families=(fam,)))
        conds.append(Condition(f"{fold}/mixed3_f0.50", "mixed3", fold,
                               PERTURBED_DIR / f"train_{fold}_mixed3.jsonl", 0.5, families=tuple(spec["seen"])))
        for frac in SWEEP_FRACTIONS:
            conds.append(Condition(f"{fold}/mixed3_f{frac:.2f}", "sweep", fold,
                                   sweep_path(fold, frac), frac, families=tuple(spec["seen"])))
        conds.append(Condition(f"{fold}/augment", "augment", fold, augment_path(fold), 0.5))
    return conds


CONDITIONS = {c.name: c for c in build_conditions()}
