"""Prepares fixed train/val/dev/test splits and generates perturbed datasets."""

import json
from pathlib import Path
from datasets import load_dataset
from src.perturbations import perturb_text

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
PERTURBED_DIR = DATA_DIR / "perturbed"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
PERTURBED_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_base_splits():
    print("Loading BoolQ dataset...")
    raw = load_dataset("google/boolq")

    # Combine available train/validation to slice into exact 4000/500/500/1000 partitions
    all_records = []
    for split_name in ["train", "validation"]:
        for row in raw[split_name]:
            all_records.append({
                "question": row["question"],
                "passage": row["passage"],
                "label": int(row["answer"])  # 1 for True, 0 for False
            })

    # Total needed = 4000 + 500 + 500 + 1000 = 6000
    if len(all_records) < 6000:
        raise ValueError(f"Not enough records: {len(all_records)} found, need 6000.")

    splits = {
        "train": all_records[:4000],
        "val": all_records[4000:4500],
        "dev": all_records[4500:5000],
        "test": all_records[5000:6000],
    }

    for name, data in splits.items():
        save_jsonl(data, SPLITS_DIR / f"{name}.jsonl")
        print(f"Saved {name}: {len(data)} examples.")

    return splits


def generate_perturbed_datasets(splits: dict):
    train_data = splits["train"]
    test_data = splits["test"]

    # Folds definition
    folds = {
        "fold_a": ["typo", "formatting", "deletion"],
        "fold_b": ["shuffle", "distractor", "synonym"],
    }

    # 1. Single-family training sets (50% noisy = 2000 clean + 2000 noisy)
    for fold_name, families in folds.items():
        for family in families:
            noisy_train = []
            # Keep first 2000 clean
            for r in train_data[:2000]:
                noisy_train.append(r)
            # Corrupt remaining 2000
            for idx, r in enumerate(train_data[2000:]):
                perturbed_q = perturb_text(r["question"], family=family, seed=10000 + idx)
                noisy_train.append({
                    "question": perturbed_q,
                    "passage": r["passage"],
                    "label": r["label"],
                    "family": family
                })
            out_path = PERTURBED_DIR / f"train_{fold_name}_{family}.jsonl"
            save_jsonl(noisy_train, out_path)
            print(f"Generated single-family training: {out_path.name}")

    # 2. Mixed-3 training sets (50% noisy = 2000 clean + ~667 per family)
    for fold_name, families in folds.items():
        mixed_train = list(train_data[:2000])  # 2000 clean
        corrupt_slice = train_data[2000:]
        
        # Partition 2000 examples into 3 chunks (~667, ~667, ~666)
        f1_slice = corrupt_slice[:667]
        f2_slice = corrupt_slice[667:1334]
        f3_slice = corrupt_slice[1334:]

        for slice_items, fam in zip([f1_slice, f2_slice, f3_slice], families):
            for idx, r in enumerate(slice_items):
                perturbed_q = perturb_text(r["question"], family=fam, seed=20000 + idx)
                mixed_train.append({
                    "question": perturbed_q,
                    "passage": r["passage"],
                    "label": r["label"],
                    "family": fam
                })

        out_path = PERTURBED_DIR / f"train_{fold_name}_mixed3.jsonl"
        save_jsonl(mixed_train, out_path)
        print(f"Generated Mixed-3 training: {out_path.name}")

    # 3. Test sets for all 6 families (1000 examples each, strictly independent seeds)
    all_families = folds["fold_a"] + folds["fold_b"]
    for fam in all_families:
        perturbed_test = []
        for idx, r in enumerate(test_data):
            # Seed 90000+ ensures zero overlap with training seed sequences
            perturbed_q = perturb_text(r["question"], family=fam, seed=90000 + idx)
            perturbed_test.append({
                "question": perturbed_q,
                "passage": r["passage"],
                "label": r["label"],
                "family": fam
            })
        out_path = PERTURBED_DIR / f"test_{fam}.jsonl"
        save_jsonl(perturbed_test, out_path)
        print(f"Generated test set: {out_path.name}")


if __name__ == "__main__":
    splits = prepare_base_splits()
    generate_perturbed_datasets(splits)