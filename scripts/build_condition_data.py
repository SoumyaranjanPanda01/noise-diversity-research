"""Builds the training files that prepare_data.py did not produce.

* Mixed-3 volume sweeps (corruption fraction 0.10, 0.25, 1.00) per fold
* Holdout-safe EDA augmentation baseline per fold (2,000 clean + 2,000 augmented)

The generator re-implements the 0.50 Mixed-3 recipe from prepare_data.py and first
verifies that it reproduces data/perturbed/train_{fold}_mixed3.jsonl exactly, so the
sweeps are guaranteed to share the primary condition's construction.
Writes data/perturbed/MANIFEST.json with sha256 + composition of every training file.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.experiment.augment import eda_augment
from src.experiment.conditions import (CONDITIONS, FOLDS, PERTURBED_DIR, SPLITS_DIR,
                                       SWEEP_FRACTIONS, augment_path, sweep_path)
from src.perturbations import perturb_text

N_TRAIN = 4000


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dumps(records: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def mixed3(train: list[dict], families: list[str], n_noisy: int) -> list[dict]:
    """Clean head + noisy tail split into 3 near-equal family chunks (same seeds as prepare_data.py)."""
    n_clean = N_TRAIN - n_noisy
    out = list(train[:n_clean])
    tail = train[n_clean:]
    a = round(n_noisy / 3)
    bounds = [(0, a), (a, 2 * a), (2 * a, n_noisy)]
    for (lo, hi), fam in zip(bounds, families):
        for idx, r in enumerate(tail[lo:hi]):
            out.append({"question": perturb_text(r["question"], family=fam, seed=20000 + idx),
                        "passage": r["passage"], "label": r["label"], "family": fam})
    return out


def augment(train: list[dict], fold: str) -> list[dict]:
    out = list(train[:2000])
    for idx, r in enumerate(train[2000:]):
        q, op = eda_augment(r["question"], fold, seed=30000 + idx)
        out.append({"question": q, "passage": r["passage"], "label": r["label"], "family": f"eda_{op}"})
    return out


def main():
    train = load_jsonl(SPLITS_DIR / "train.jsonl")
    assert len(train) == N_TRAIN

    for fold, spec in FOLDS.items():
        ref = (PERTURBED_DIR / f"train_{fold}_mixed3.jsonl").read_text(encoding="utf-8")
        if dumps(mixed3(train, spec["seen"], 2000)) != ref:
            raise SystemExit(f"Generator does not reproduce existing {fold} mixed3 file; aborting.")
        print(f"[ok] generator reproduces train_{fold}_mixed3.jsonl byte-for-byte")

        for frac in SWEEP_FRACTIONS:
            recs = mixed3(train, spec["seen"], round(frac * N_TRAIN))
            sweep_path(fold, frac).write_text(dumps(recs), encoding="utf-8")
            print(f"wrote {sweep_path(fold, frac).name}")
        augment_path(fold).write_text(dumps(augment(train, fold)), encoding="utf-8")
        print(f"wrote {augment_path(fold).name}")

    manifest = {}
    for c in CONDITIONS.values():
        raw = c.data_path.read_bytes()
        recs = [json.loads(l) for l in raw.decode("utf-8").splitlines()]
        manifest[c.name] = {
            "path": c.data_path.as_posix(),
            "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),  # EOL-agnostic
            "n": len(recs),
            "composition": dict(Counter(r.get("family", "clean") for r in recs)),
        }
        assert len(recs) == N_TRAIN, c.name
    (PERTURBED_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote data/perturbed/MANIFEST.json")


if __name__ == "__main__":
    main()
