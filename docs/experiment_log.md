# Experiment log

Continues the lab notebook kept outside the repo (`noise-diversity.docx`, not tracked: protocol,
perturbation families, data splits, quality gate, untuned-baseline script). Newest entries at the bottom.

## 2026-11-04 — Training/eval/analysis code; data reproducibility fix

**State found.** Data splits and perturbed files were done; `eval_baseline.py` existed but had not been run
(`results/` was empty); no training code; `tests/test_semantic_perturbations.py` failed at import
(`DISTRACTOR_POOL` was renamed to `NEUTRAL_DISTRACTORS`).

**Bug: synonym perturbation not reproducible.** `get_wordnet_synonyms` returned `list(set)`, and set order
changes with `PYTHONHASHSEED`, so a fixed seed still gave different synonyms in each process.
Regenerating differed on 661/1000 `test_synonym` items, 1,288/2,000 noisy `train_fold_b_synonym` items
and 420 `train_fold_b_mixed3` items. Fixed (`sorted`), added a cross-process regression test, and regenerated.
Only those 3 files changed; the other families and the splits are byte-identical. No model had seen the old files. → D3.

**New conditions data** (`scripts/build_condition_data.py`). The generator first verified that it reproduces both
existing 0.50 Mixed-3 files byte-for-byte, then wrote the sweeps (0.10 / 0.25 / 1.00) and the holdout-safe EDA
baseline (→ D2). `data/perturbed/MANIFEST.json` records sha256 + composition; all 18 conditions = 4,000 examples.

**Design decisions** (→ `docs/deviations.md`): NEFTune shared across folds (D1, 54 runs), logit-based Yes/No
scoring (D4), answer-only loss + passage truncation (D5; 15/4,000 truncated).

**Code.**
* `src/experiment/conditions.py` — condition registry
* `src/experiment/modeling.py` — encoder, left-padded batches with only the last positions sent through the LM head, LoRA loop, NEFTune hook, Yes/No scorer
* `scripts/train.py` — one run, full run records
* `scripts/run_all.py` — resumable matrix with sharding
* `scripts/analyze.py` — paired bootstrap, sweep, transfer matrix, figures
* `notebooks/run_experiments_gpu.ipynb` — GPU runner

**Validation.**
* `pytest`: 16 passed.
* `analyze.py` on *synthetic* runs (scratch dir, never in `results/`): a planted +3 pp effect was recovered
  (+2.97 / +3.49 pp, CI excludes 0), and a planted 0 effect came out null.
* CPU smoke run (1 step, 8 eval items) completed train → val → 7 test sets → records. The step-1 loss
  (0.6189) was identical across two invocations (deterministic seeding).

**Compute.** No CUDA on the dev laptop (Ryzen 5 5500U, 14 GB). Measured ~155 s per optimizer step
(the CPU was shared with another job) → ~14 h per run, about a month for the matrix. Decision: run on a
free cloud GPU via the notebook (D6).

**Next.** Push → run the notebook (GPU smoke cell first) → copy `results/` back → `analyze.py` → write up `docs/results.md`.

## 2026-11-04 — GPU smoke test: CUDA OOM at micro-batch 16

The first Colab T4 smoke run (`fold_a/mixed3_f0.50`, `--max-steps 5`) failed with `torch.OutOfMemoryError`
in the first forward pass: 14.0 GiB allocated by PyTorch on the 14.56 GiB T4 at micro-batch 16 (padded
sequences up to ~500 tokens). Fix: the default micro-batch is now 4 on all devices, with gradient
accumulation ×4 to keep the protocol's effective batch of 16. The loss is already normalised by the token
count of the full 16-example batch, so the accumulated gradient equals the full-batch gradient (protocol
unchanged). Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. No results were affected:
smoke runs are excluded from the analysis.
