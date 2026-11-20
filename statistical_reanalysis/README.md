# Post-hoc statistical reanalysis (CPU only)

Everything in this folder was added on 2026-11-20, **after** all test results were known. None of it
replaces the pre-registered analysis (`scripts/analyze.py`, `results/analysis/`), which is left untouched.
No model was trained or evaluated for this reanalysis: every number is computed from the per-item
predictions already stored in `results/runs/*/predictions.json` and the frozen data in `data/`.

## Run

From this folder, with numpy, scipy and matplotlib installed (`pip install -r ../requirements.txt`):

```sh
sh run_all.sh          # or run the five scripts below in order; about 2 minutes on a laptop CPU
```

| script | what it does | outputs (in `outputs/` unless stated) |
|---|---|---|
| `01_verify_records.py` | integrity of all 55 run records; accuracy recomputed from predictions; data hashes vs MANIFEST; re-runs `scripts/analyze.py` and checks byte-identity; counts exact p(Yes) = 0.5 ties; writes SHA-256 of every original record | `01_record_integrity.{md,json}`, `original_records_sha256.txt` |
| `02_passage_overlap.py` | train/test passage overlap (verbatim and near-duplicate, 5-gram Jaccard >= 0.5); passage clusters for the cluster bootstrap | `02_passage_overlap.{md,json}`, `test_item_flags.csv` |
| `03_contrasts_reanalysis.py` | primary and 14 secondary contrasts under 7 settings (original item bootstrap; passage-cluster bootstrap; two overlap-excluded subsets; two tie rules; threshold-free p(gold)); seed-level t intervals, combined item+seed intervals, two-stage bootstrap, leave-one-seed-out, per-family breakdown, Holm adjustment; transfer matrix with CIs | `03_contrasts_reanalysis.{md,json}` |
| `04_distractor_bias.py` | distractor answer-preference analysis for all 55 runs: confusion matrices, Yes-rate shifts with CIs, AUC, label-free re-thresholding, flips, per-sentence effects, within-sentence AUC, dose-response, across-run association | `04_distractor_bias.{md,json}`, `04_distractor_per_run.csv` |
| `05_figures.py` | the three figures used in the paper (vector PDF, embedded TrueType fonts) | `../revised_manuscript/figures/*.pdf` |

`common.py` holds the shared loaders and the bootstrap engine. Bootstrap draws use fixed seeds
(`12345`, as in `scripts/analyze.py`, and `777` for the two-stage bootstrap), so outputs are byte-reproducible.
Setting S0 in `03` reproduces the original `results/analysis/summary.md` numbers exactly; this is the
check that the reanalysis code computes the same quantity as the original.

## Definitions

* **Passage cluster**: test items whose passage text is identical, merged with near-duplicate test passages
  (5-gram-shingle Jaccard >= 0.5) by union-find. 966 clusters for 1,000 items.
* **Cluster bootstrap**: resample clusters with replacement, keep all items of each drawn cluster, apply the
  same draw to every condition, family, seed and fold (pairing preserved); effect = ratio of summed item
  differences to summed cluster sizes.
* **Combined interval** (approximate): `effect +- 1.96 * sqrt(SE_items^2 + SD_seed^2 / n_seeds)`, where
  SE_items is the cluster-bootstrap SE of the seed-averaged effect and SD_seed the SD of the three per-seed
  effects. Treat it as a rough guide: with three seeds the seed SD is itself very uncertain.
* **Two-stage bootstrap**: per draw, resample each condition's seeds with replacement (independently across
  conditions, so seed pairing is dropped, which is conservative) and resample clusters jointly.
* **Label-free re-threshold** (distractor analysis): the threshold on distractor-input probabilities that
  reproduces the same model's predicted-Yes rate on the clean versions of the same questions. It uses test
  inputs but no labels; it is a diagnostic, not a deployable method.

## What could not be done with the available artifacts

* Passage/title identifiers are not stored in the splits, so passage identity is inferred from text.
* LoRA adapters are not in the repository (git-ignored, not on this machine), so no mechanistic analysis
  of the trained models (for example attention to the distractor) is possible.
* Only three seeds exist; seed-level uncertainty cannot be estimated precisely, and no extra seeds can be run.
* No validation-set version of the perturbed families exists with model predictions, so any threshold or
  calibration analysis has to be done transductively on test inputs (see definition above).
