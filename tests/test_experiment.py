"""Tests for the condition registry and the holdout-safe augmentation baseline."""

from src.experiment.augment import FOLD_OPS, eda_augment
from src.experiment.conditions import CONDITIONS, FOLDS


def test_condition_count():
    # 2 shared (clean, neftune) + 8 per fold (3 single, mixed3@0.5, 3 sweeps, augment)
    assert len(CONDITIONS) == 18
    assert sum(c.fold is None for c in CONDITIONS.values()) == 2


def test_folds_are_disjoint_and_complementary():
    for spec in FOLDS.values():
        assert not set(spec["seen"]) & set(spec["unseen"])
    assert FOLDS["fold_a"]["seen"] == FOLDS["fold_b"]["unseen"]


def test_augment_ops_avoid_unseen_families():
    op_family = {"rd": "deletion", "rs": "shuffle", "sr": "synonym"}
    for fold, ops in FOLD_OPS.items():
        for op in ops:
            assert op_family.get(op) not in FOLDS[fold]["unseen"]


def test_augment_deterministic_and_label_free():
    q = "is the eiffel tower taller than the statue of liberty"
    for fold in FOLD_OPS:
        assert eda_augment(q, fold, seed=3) == eda_augment(q, fold, seed=3)
        assert eda_augment("", fold, seed=3)[0] == ""
