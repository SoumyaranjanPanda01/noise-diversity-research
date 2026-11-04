# Protocol deviations & implementation decisions (vs. docs/protocol_v1.md)

Every change to the frozen protocol is logged here *before* test results are seen.
Status as of 2026-11-04: **no test-set results exist yet**; all decisions below were made
blind to test performance.

| ID | Date | Change | Reason | Effect on design |
|---|---|---|---|---|
| D1 | 2026-11-04 | NEFTune baseline is a single **shared** condition (clean data + NEFTune, α=5), like the clean baseline, instead of one per fold. | NEFTune on clean data is fold-independent: the two "per-fold" conditions would be the same configuration trained twice. | 18 conditions × 3 seeds = **54 training runs** (+1 untuned eval) instead of 19 × 3 = 57. |
| D2 | 2026-11-04 | EDA augmentation baseline uses only the EDA operations that do **not** correspond to the fold's unseen families: fold A → random deletion + random insertion; fold B → random swap + random insertion + synonym replacement. Random insertion copies a word already in the question (no new content). α=0.1, one op per example, 2,000 augmented + 2,000 clean. | Standard EDA (SR/RI/RS/RD) contains synonym, swap and deletion edits, which would leak held-out families and break the family-level holdout (§2). | Augmentation baseline is holdout-clean; matched corrupted size (2,000) as specified. |
| D3 | 2026-11-04 | Fixed non-determinism in `apply_synonym` (`get_wordnet_synonyms` returned `list(set)`, whose order depends on `PYTHONHASHSEED`). Now `sorted(...)`. Regenerated data with `scripts/prepare_data.py`. | Synonym files were not reproducible from code + seed (661/1000 test items differed on regeneration). | Only `test_synonym`, `train_fold_b_synonym`, `train_fold_b_mixed3` changed; all other files byte-identical. No model had been trained or evaluated on the old files. Regression test added. |
| D4 | 2026-11-04 | Evaluation scores **p(Yes) vs p(No)** from the next-token logits after the chat generation prompt, instead of greedy generation + string parsing (`eval_baseline.py`). | Deterministic, no unparseable outputs, ~10× cheaper; gives per-item probabilities for paired statistics. Same prompt as `eval_baseline.py`. | Exact-match accuracy = 1[p(Yes) > 0.5] == label. Applied identically to every model, including the untuned baseline (`--condition untuned`). |
| D5 | 2026-11-04 | Training loss is computed only on the answer token and `<|im_end|>` (prompt tokens masked). Passages are truncated (never the question) so the sequence fits 512 tokens. | Standard SFT target masking; the 512 cap is protocol §3. | 15 / 4,000 train examples need passage truncation. |
| D6 | 2026-11-04 | Training runs on a cloud GPU (Colab T4) via `notebooks/run_experiments_gpu.ipynb`. | The development laptop has no CUDA GPU; a full run measured ~14 h on its CPU (≈1 month for the matrix). | Mixed precision (fp16 autocast + grad scaling on T4, bf16 where supported) with fp32 master weights. Environment recorded per run. |

## Decisions that are *not* deviations (clarifications)

* **Seeds** {0, 1, 2} control LoRA init, dropout and data order. Perturbed data is fixed (pre-generated) across seeds.
* **Steps are matched**: every condition has 4,000 examples → 250 steps/epoch × 2 = 500 optimizer steps.
* **Sweeps** reuse the 0.50 Mixed-3 construction (`scripts/build_condition_data.py` verifies it reproduces the existing 0.50 files byte-for-byte): clean head `train[:4000-N]`, noisy tail split into thirds.
* **Validation set** (500) is evaluated after each epoch for monitoring only; no checkpoint selection (the final model is evaluated). The dev set is not used, since no tuning was done.
* **Test set** is evaluated exactly once per trained model, after training finishes.
* **Statistics** (`scripts/analyze.py`): paired bootstrap (10,000 resamples) over the 1,000 test questions, jointly across families, seeds and conditions; "meaningful" = 95% CI excludes 0 **and** |effect| > SD of the per-seed effect.
