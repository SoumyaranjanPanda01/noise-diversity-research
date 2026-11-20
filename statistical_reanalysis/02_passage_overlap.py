"""Step 2: train/test passage overlap and test-set passage clusters.

Identification: passage TEXT (the BoolQ copy used here has no passage/title ids).
* exact: the test item's passage string occurs verbatim in data/splits/train.jsonl
* near : exact, or max 5-gram-shingle Jaccard similarity with some train passage >= 0.5
* cluster: test items sharing an identical passage, merged with near-duplicate (Jaccard >= 0.5)
  test passages (union-find). Used by the cluster bootstrap in 03.

Output: outputs/test_item_flags.csv (one row per test item), outputs/02_passage_overlap.{md,json}

    python statistical_reanalysis/02_passage_overlap.py
"""

import csv
from collections import Counter

from common import OUT, ROOT, load_jsonl, passage_flags, write


def main():
    rows = passage_flags()
    with open(OUT / "test_item_flags.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    test = load_jsonl(ROOT / "data/splits/test.jsonl")
    train = load_jsonl(ROOT / "data/splits/train.jsonl")
    n_distinct = len({r["passage"] for r in test})
    exact_items = sum(r["train_passage_exact"] for r in rows)
    near_items = sum(r["train_passage_near"] for r in rows)
    exact_pass = len({t["passage"] for t, r in zip(test, rows) if r["train_passage_exact"]})
    near_pass = len({t["passage"] for t, r in zip(test, rows) if r["train_passage_near"]})
    sizes = Counter(Counter(r["cluster"] for r in rows).values())
    n_clusters = len({r["cluster"] for r in rows})
    exact_pass_sizes = Counter(Counter(r["passage"] for r in test).values())
    yes = lambda sel: sum(r["label"] for r in sel) / max(1, len(sel))
    # how many train questions share the overlapping passages
    tr_count = Counter(r["passage"] for r in train)
    n_train_q_on_overlap = sum(tr_count[p] for p in {t["passage"] for t, r in zip(test, rows) if r["train_passage_exact"]})
    res = {
        "test_items": len(rows), "distinct_test_passages": n_distinct,
        "items_with_exact_train_passage": exact_items, "distinct_passages_exact_in_train": exact_pass,
        "items_with_near_train_passage": near_items, "distinct_passages_near_in_train": near_pass,
        "train_questions_on_exactly_overlapping_passages": n_train_q_on_overlap,
        "test_questions_in_train": sum(r["question"] in {x["question"] for x in train} for r in test),
        "n_clusters": n_clusters, "cluster_size_counts": dict(sorted(sizes.items())),
        "exact_passage_group_size_counts": dict(sorted(exact_pass_sizes.items())),
        "gold_yes_rate": {"all": yes(rows), "exact_overlap": yes([r for r in rows if r["train_passage_exact"]]),
                          "no_exact_overlap": yes([r for r in rows if not r["train_passage_exact"]]),
                          "no_near_overlap": yes([r for r in rows if not r["train_passage_near"]])},
    }
    lines = ["# 02 Passage overlap and clusters (auto-generated)", "",
             "Passage identity is established from the passage text (no ids are stored in the splits).", "",
             "| quantity | value |", "|---|---|",
             f"| test items | {len(rows)} |",
             f"| distinct test passages | {n_distinct} |",
             f"| test items whose passage occurs verbatim in train | **{exact_items}** |",
             f"| distinct test passages occurring verbatim in train | {exact_pass} |",
             f"| test items whose passage is verbatim or near-duplicate (5-gram Jaccard >= 0.5) of a train passage | **{near_items}** |",
             f"| distinct test passages, verbatim or near-duplicate in train | {near_pass} |",
             f"| train questions attached to the verbatim-overlapping passages | {n_train_q_on_overlap} |",
             f"| test questions occurring verbatim in train | {res['test_questions_in_train']} |",
             f"| passage clusters in test (identical + near-duplicate merged) | {n_clusters} |",
             f"| cluster size counts {{size: n}} | {res['cluster_size_counts']} |",
             f"| gold Yes rate: all / exact-overlap / no exact overlap / no near overlap | "
             f"{yes(rows):.3f} / {res['gold_yes_rate']['exact_overlap']:.3f} / "
             f"{res['gold_yes_rate']['no_exact_overlap']:.3f} / {res['gold_yes_rate']['no_near_overlap']:.3f} |",
             "",
             "Correction to the manuscript: \"136 of the 1,000 test passages also occur in the training split\" should read",
             f"\"{exact_items} of the 1,000 test questions ({exact_pass} of {n_distinct} distinct test passages)\".",
             f"Counting near-duplicates (same Wikipedia paragraph with small edits), {near_items} test questions are affected."]
    write("02_passage_overlap", "\n".join(lines) + "\n", res)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
