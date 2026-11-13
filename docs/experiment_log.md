# Experiment log

Continues the lab notebook kept outside the repo (`noise-diversity.docx`, not tracked: protocol,
perturbation families, data splits, quality gate, untuned-baseline script). Newest entries at the bottom.

## Training/eval/analysis code; data reproducibility fix

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

## GPU smoke test: CUDA OOM at micro-batch 16

The first Colab T4 smoke run (`fold_a/mixed3_f0.50`, `--max-steps 5`) failed with `torch.OutOfMemoryError`
in the first forward pass: 14.0 GiB allocated by PyTorch on the 14.56 GiB T4 at micro-batch 16 (padded
sequences up to ~500 tokens). Fix: the default micro-batch is now 4 on all devices, with gradient
accumulation ×4 to keep the protocol's effective batch of 16. The loss is already normalised by the token
count of the full 16-example batch, so the accumulated gradient equals the full-batch gradient (protocol
unchanged). Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. No results were affected:
smoke runs are excluded from the analysis.

## Notebook dependency fix; evidence table verified

**Notebook.** `torchao>=0.16.0` was added to the notebook's pip install cell. The other pinned versions
(`transformers==5.21.0`, `peft==0.21.2`) are unchanged, and the cell sources were re-saved in Colab's
list-of-lines format (no code changes). `COLAB_RESUME_GUIDE.pdf` (a personal resume guide) is local only and
listed in `.gitignore`.

**Evidence table** (`docs/evidence_table.md`). All six entries were checked against the original arXiv PDFs.
14 claims in the earlier draft were corrected (listed under the table). Main corrections: [1] uses four base
models and six instruction-only perturbation types, and sweeps the perturbed fraction at a fixed dataset size
(volume "Partially" controlled); [2] has no noisy-task-SFT baseline and its perturbations are adversarial;
[3] evaluates on the clean test set only; [6] trains single types only. A synthesis section was added: no
paper varies diversity at a fixed corruption volume, and no text paper tests on held-out perturbation
families, so a null result from our study would still be informative. Downloaded PDFs are kept in
`references/` (git-ignored).

**State.** No training results yet (`results/runs/` is empty).

**Next.** Run the notebook matrix on Colab → copy `results/` back → `analyze.py` → write up `docs/results.md`.

## 2026-11-14: Full matrix complete; results analysed

**Runs.** All 55 runs (18 conditions × 3 seeds + untuned) completed on Colab T4 between 2026-11-04 and
2026-11-13. Records were downloaded as `results_export.zip` and unpacked into `results/`. Checks: every
training-data sha256 matches `MANIFEST.json`; all runs have 500 steps and micro-batch 4; code (`src/`, `scripts/`,
`configs/`) is identical across the three commits the runs were made at; every `train_log.jsonl` has steps
1 to 500 in order. Three runs were executed twice after session resumes (duplicate registry rows) and gave identical
accuracies both times. The `__gpusmoke` debug run is excluded.

**Export bug.** The notebook's last cell wrote the zip to `../results_export.zip` from inside `results/`, which
is a symlink to Google Drive, so the zip landed on the Drive FUSE mount and failed ("zip I/O error: Operation
not supported"). Fixed to write to `/content/results_export.zip`.

**Analysis.** `analyze.py` on Colab and locally gave identical `summary.md` / `summary.json`. Primary result is
null (pooled −0.14 pp, CI [−0.59, +0.31]). Full write-up: `docs/results.md`.

**Next.** arXiv paper draft.

## 2026-11-14: Distractor diagnostic; paper draft

**Exploratory diagnostic** (`scripts/diag_distractor.py`, not pre-registered). The distractor drop is mostly an
answer bias towards "No": the untuned model predicts "Yes" on 0.8% of distractor items (37.1% on clean), and the clean
model on 38.8% (64.7% on clean) against a gold rate of 63.4%. Models trained with distractor examples, including fold B
Mixed-3 with only 667 of them, are back at their clean rate. Added to `docs/results.md`.

**Data facts for the paper.** Share of test questions actually changed by each family: typo 80.0%, formatting 98.1%,
deletion 77.5%, shuffle 82.3%, distractor 100%, synonym 79.7% (training sets are similar). Mean run time on T4:
1,973 s training + 584 s evaluation.

**Paper.** First full LaTeX draft at `paper/main.tex`: single file, bibliography embedded, figures drawn in pgfplots
(no image files), appendix after the references, no LaTeX comments, `\typeout` line after `\end{document}`, as the
arXiv upload checklist requires. Not compiled yet (no TeX installation on the dev laptop).
