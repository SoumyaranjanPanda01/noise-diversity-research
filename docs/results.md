# Results: Noise Diversity vs. Corruption Volume

All numbers below come from `scripts/analyze.py` (output: `results/analysis/summary.md`, `summary.json`).
The analysis was run on Colab and re-run locally on the downloaded records, and both runs gave identical
output. Accuracies are on the 1,000-item test set, evaluated once per trained model. "pp" = percentage points.
Statistical rule (protocol §1, fixed before any test result was seen): an effect is **meaningful** only if the
paired-bootstrap 95% CI (10,000 resamples over test items, jointly over seeds) excludes 0 **and** |effect|
exceeds the cross-seed SD of the effect.

## Run inventory and integrity

* **55 / 55 runs complete**: 18 conditions × 3 seeds + 1 untuned evaluation. Every run has 500 optimizer
  steps, micro-batch 4 × 4 accumulation (effective batch 16), and ran on a CUDA T4.
* **Training data**: every run's `train_data_sha256_lf` matches `data/perturbed/MANIFEST.json`.
* **Code**: runs were made at three commits (`4962e62`: 4 runs, `3070f53`: 8, `b30db8a`: 43). `src/`, `scripts/`
  and `configs/` are identical across all three. The commits differ only in docs, `.gitignore` and the
  notebook.
* **Excluded debug run**: `fold_a__mixed3_f0.50__s0__gpusmoke` (5 steps, 32 eval items, tagged) was the GPU
  smoke test. The analysis excludes it automatically and it is git-ignored.
* **Repeated runs**: after Colab session resumes, three runs were executed twice (`fold_b/mixed3_f0.25`
  seeds 0 and 1, `fold_b/mixed3_f1.00` seed 1). Each has two rows in `run_registry.csv`, and the run directory
  holds the second execution. In all three cases the two executions gave **identical** validation and
  test accuracies on every eval set. This is direct evidence that training and evaluation are deterministic
  given the seed. The two executions of `fold_b/mixed3_f0.25` seed 0 overlapped in time, and the first
  execution's registry row has an empty `final_loss`. The kept directory's `train_log.jsonl` is complete
  (steps 1 to 500, in order). Every run directory has a complete, in-order 500-step log.

## Primary result: diversity does not help on unseen families (null)

Mixed-3 at 50% corruption minus the mean of the three single-family models, on the fold's **unseen** families:

| fold (unseen families) | Mixed-3 | mean single | effect | 95% CI | seed SD | verdict |
|---|---|---|---|---|---|---|
| A (shuffle, distractor, synonym) | 69.06 | 69.76 | −0.71 pp | [−1.38, −0.05] | 1.11 | not meaningful (CI excludes 0, but \|effect\| ≤ seed SD) |
| B (typo, formatting, deletion) | 77.18 | 76.76 | +0.42 pp | [−0.24, +1.08] | 0.48 | null |
| pooled | | | −0.14 pp | [−0.59, +0.31] | 0.38 | null |

Per-seed effects: fold A −1.17, −1.51, +0.56; fold B +0.88, +0.47, −0.08 pp. The direction is not consistent
across seeds in either fold.

**Answer to the research question:** at a fixed 50% corruption volume, training on a mixture of three
perturbation families did **not** improve accuracy on held-out perturbation families compared with training on a
single family. The pooled estimate is −0.14 pp, and the CI rules out a pooled benefit larger than about +0.3 pp.

## Secondary contrasts (Mixed-3 at 0.50 minus X)

| fold | comparison | test sets | effect | 95% CI | seed SD | verdict |
|---|---|---|---|---|---|---|
| A | clean baseline | unseen | −2.29 | [−3.14, −1.48] | 1.03 | **meaningful** |
| A | NEFTune | unseen | −2.06 | [−2.92, −1.17] | 0.20 | **meaningful** |
| A | EDA augmentation | unseen | −1.24 | [−2.07, −0.40] | 0.50 | **meaningful** |
| A | best single (formatting) | unseen | −1.01 | [−1.78, −0.27] | 1.48 | not meaningful |
| A | mean single | seen | −0.32 | [−1.00, +0.35] | 1.00 | null |
| A | mean single | clean | −0.74 | [−1.63, +0.12] | 0.62 | null |
| A | clean baseline | clean | −0.57 | [−1.70, +0.53] | 0.47 | null |
| B | clean baseline | unseen | +0.37 | [−0.43, +1.17] | 0.57 | null |
| B | NEFTune | unseen | +0.06 | [−0.77, +0.88] | 0.78 | null |
| B | EDA augmentation | unseen | −0.32 | [−1.11, +0.48] | 0.58 | null |
| B | best single (distractor) | unseen | +0.20 | [−0.54, +0.96] | 0.60 | null |
| B | mean single | seen | +4.51 | [+3.61, +5.40] | 0.88 | **meaningful** (see note) |
| B | mean single | clean | +0.83 | [−0.02, +1.67] | 0.62 | null |
| B | clean baseline | clean | +0.70 | [−0.37, +1.77] | 1.21 | null |

These 14 contrasts are not corrected for multiple comparisons. Treat them as supporting evidence, not
confirmatory tests.

**Fold A: perturbed training costs about 1 to 2 pp on unseen families.** Mixed-3 is meaningfully worse than
the clean baseline, NEFTune and EDA on fold A's unseen families. On clean test data it is not distinguishable
from the clean baseline (−0.57 pp, null), so the cost is specific to perturbed inputs. Descriptively, the drop
against the clean baseline shows up in all three unseen families (shuffle 76.3 vs 78.1, distractor 57.5 vs
60.8, synonym 73.3 vs 75.1), so distractor alone does not explain it. The three fold A single-family
models also sit below the clean baseline on these families (transfer matrix, rows typo/formatting/deletion).
In fold B no such cost appears: Mixed-3 is null against every baseline.

**Note on fold B "seen" (+4.51 pp).** This contrast is real but follows from the design, not from
diversity. Only one of fold B's three single-family models was trained on distractor, and distractor is the
only family where training produces a large gain (next section). The other two models score about 60% on
distractor, which pulls the single-family mean on seen families down. Mixed-3 includes distractor
examples, so it does not have that gap.

## Volume sweep (Mixed-3, unseen families, mean ± SD over seeds)

| fold | 0.00 (clean) | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| A | 71.34 ± 0.50 | 70.76 ± 0.57 | 70.90 ± 1.76 | 69.06 ± 1.15 | 69.18 ± 0.21 |
| B | 76.81 ± 0.23 | 76.92 ± 0.93 | 76.77 ± 0.20 | 77.18 ± 0.54 | 76.51 ± 0.44 |

Descriptive only: the protocol did not specify a test for the sweep. In fold A, unseen accuracy is about
2 pp lower at 0.50 and 1.00 than with clean training. In fold B it is flat within about 0.7 pp at every
fraction. More corruption volume does not buy unseen-family robustness in either fold.
Figure: `results/analysis/volume_sweep.png`.

## Transfer between families (single-family model minus clean baseline, pp)

| trained on \ tested on | typo | formatting | deletion | shuffle | distractor | synonym |
|---|---|---|---|---|---|---|
| typo | **−0.7** | −0.9 | −1.3 | −0.8 | −2.1 | −1.9 |
| formatting | +0.1 | **+0.3** | +0.5 | −0.3 | −2.6 | −0.9 |
| deletion | −0.2 | −0.5 | **−0.5** | −1.2 | −2.6 | −1.8 |
| shuffle | +0.2 | −1.1 | −0.2 | **−0.3** | −1.4 | −0.6 |
| distractor | −0.4 | +0.3 | +0.6 | −1.3 | **+18.1** | −0.1 |
| synonym | +0.0 | −0.1 | +0.3 | −1.0 | +1.4 | **+0.8** |

* **Distractor is the only family that clean fine-tuning leaves unsolved.** The untuned model scores 37.4% on
  distractor items, clean fine-tuning raises that to 60.8%, and training on distractor raises it to
  78.9% (+18.1 pp, back to clean-test level). No other training family transfers to it: the off-diagonal
  entries in the distractor column range from −2.6 to +1.4.
* **The other five families barely hurt a clean-trained model.** The clean baseline loses at most 3.7 pp
  on them relative to clean test data (78.8%; worst is synonym at 75.1%). Even training on the family itself (diagonal) changes accuracy
  by less than 1 pp, so there is little room for transfer to show.
* The cells in this table are descriptive means. No CIs are computed for them.

## Exploratory: the distractor gap is an answer bias

Not pre-registered. Source: `scripts/diag_distractor.py` → `results/analysis/distractor_bias.{md,json}`.
The gold "Yes" rate on the test set is 63.4%.

| condition | test set | predicted Yes | acc on gold Yes | acc on gold No | acc |
|---|---|---|---|---|---|
| untuned | clean | 37.1 | 48.1 | 82.0 | 60.5 |
| untuned | distractor | 0.8 | 1.3 | 100.0 | 37.4 |
| clean | clean | 64.7 ± 0.8 | 84.3 ± 0.5 | 69.3 ± 1.9 | 78.8 ± 0.7 |
| clean | distractor | 38.8 ± 6.6 | 49.7 ± 7.2 | 80.1 ± 6.5 | 60.8 ± 3.2 |
| fold A Mixed-3 | distractor | 31.1 ± 6.7 | 41.0 ± 7.7 | 86.2 ± 5.6 | 57.5 ± 3.2 |
| fold B Mixed-3 | distractor | 65.1 ± 1.7 | 84.8 ± 1.1 | 69.1 ± 3.0 | 79.1 ± 0.8 |
| fold B single distractor | distractor | 65.3 ± 1.3 | 84.9 ± 1.4 | 68.6 ± 1.7 | 78.9 ± 0.8 |

An appended distractor sentence mainly pushes the model towards answering "No". The untuned model almost never
says "Yes", and clean fine-tuning reduces but does not remove the shift. The large seed SD on this family tracks how
large the shift is. Any model that saw distractor examples in training, including fold B's Mixed-3 with only 667 of
them, answers "Yes" at its clean-input rate. Caveat: the distractor test and training sets draw from the same pool of
8 sentences, so the in-family gain may partly reflect familiarity with those sentences.

## Summary

1. **Primary hypothesis: not supported.** Mixing three perturbation families does not beat single-family
   training on unseen families at a fixed corruption volume (pooled −0.14 pp, CI [−0.59, +0.31]).
2. **Perturbed training can cost a little on unseen families.** In fold A, Mixed-3 is 1 to 2 pp worse than
   clean, NEFTune and EDA training (meaningful by the protocol rule). This is not seen in fold B.
3. **Robustness is family-specific.** The one family with a large robustness gap (distractor) is closed only
   by training on that family. Nothing transfers to it, and the remaining families leave little gap to close.

## Limitations

* One model (Qwen2.5-0.5B-Instruct), one task (BoolQ) and three seeds. The 1.5B scale check (protocol §3,
  optional) was not run.
* The two folds are not symmetric: fold A's unseen set contains distractor (hard, high-variance, seed SD up to
  5.3 pp), while fold B's unseen families are nearly solved by clean training. Fold B therefore had little
  headroom to show any effect.
* The five non-distractor families produce small accuracy drops, which limits the power of any
  diversity-vs-single comparison on them.
* Secondary contrasts are uncorrected for multiplicity, and the sweep has no inferential test.
