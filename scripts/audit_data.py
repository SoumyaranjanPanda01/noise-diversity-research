"""Data Quality Gate: checks split leakage, length stats, and samples edits."""

import json
from pathlib import Path

PERTURBED_DIR = Path("data/perturbed")
SPLITS_DIR = Path("data/splits")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_quality_gate():
    print("=== 1. Checking Data Leakage between Train and Test ===")
    train_records = load_jsonl(SPLITS_DIR / "train.jsonl")
    test_records = load_jsonl(SPLITS_DIR / "test.jsonl")

    train_qs = {r["question"].strip().lower() for r in train_records}
    test_qs = {r["question"].strip().lower() for r in test_records}
    overlap = train_qs.intersection(test_qs)

    print(f"Train unique questions: {len(train_qs)}")
    print(f"Test unique questions:  {len(test_qs)}")
    print(f"Overlap count: {len(overlap)}")
    if overlap:
        print("WARNING: Leakage detected between train and test splits!")
    else:
        print("PASSED: Zero leakage between train and test splits.")

    print("\n=== 2. Perturbation Audit & Length Checks ===")
    families = ["typo", "formatting", "deletion", "shuffle", "distractor", "synonym"]

    for fam in families:
        test_file = PERTURBED_DIR / f"test_{fam}.jsonl"
        perturbed_data = load_jsonl(test_file)

        len_ratios = []
        for orig, pert in zip(test_records, perturbed_data):
            orig_len = max(len(orig["question"].split()), 1)
            pert_len = len(pert["question"].split())
            len_ratios.append(pert_len / orig_len)

        avg_ratio = sum(len_ratios) / len(len_ratios)
        print(f"[{fam.upper()}] Avg word ratio (perturbed / clean): {avg_ratio:.2f}")

    print("\n=== 3. Spot-Checking 2 Examples Per Family ===")
    for fam in families:
        test_file = PERTURBED_DIR / f"test_{fam}.jsonl"
        perturbed_data = load_jsonl(test_file)
        print(f"\n--- Family: {fam} ---")
        for i in range(2):
            clean = test_records[i]["question"]
            pert = perturbed_data[i]["question"]
            print(f"Clean:     {clean}")
            print(f"Perturbed: {pert}")


if __name__ == "__main__":
    run_quality_gate()