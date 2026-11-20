# 02 Passage overlap and clusters (auto-generated)

Passage identity is established from the passage text (no ids are stored in the splits).

| quantity | value |
|---|---|
| test items | 1000 |
| distinct test passages | 969 |
| test items whose passage occurs verbatim in train | **136** |
| distinct test passages occurring verbatim in train | 121 |
| test items whose passage is verbatim or near-duplicate (5-gram Jaccard >= 0.5) of a train passage | **173** |
| distinct test passages, verbatim or near-duplicate in train | 154 |
| train questions attached to the verbatim-overlapping passages | 208 |
| test questions occurring verbatim in train | 0 |
| passage clusters in test (identical + near-duplicate merged) | 966 |
| cluster size counts {size: n} | {1: 935, 2: 29, 3: 1, 4: 1} |
| gold Yes rate: all / exact-overlap / no exact overlap / no near overlap | 0.634 / 0.662 / 0.630 / 0.631 |

Correction to the manuscript: "136 of the 1,000 test passages also occur in the training split" should read
"136 of the 1,000 test questions (121 of 969 distinct test passages)".
Counting near-duplicates (same Wikipedia paragraph with small edits), 173 test questions are affected.
