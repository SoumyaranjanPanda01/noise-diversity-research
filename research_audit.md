# Research audit: "Does Perturbation Diversity Buy Robustness to Unseen Noise?"

Audit date: 2026-11-20. Scope: methodology, implementation, run records, statistics, manuscript claims, and
submission readiness. Constraint honoured throughout: **no model was trained or evaluated**. Every new number
comes from CPU analysis of the existing per-item predictions (`results/runs/*/predictions.json`) and frozen data.
All post-hoc code and outputs are in `statistical_reanalysis/`. Original records are unchanged (SHA-256 list in
`statistical_reanalysis/outputs/original_records_sha256.txt`, re-verified after every run).

## 1. Artifact inventory

| artifact | location | status |
|---|---|---|
| Protocol (pre-registered sketch) | `docs/protocol_v1.md` | read; 57-run plan, decision rule |
| Deviations D1 to D6 (logged before test results) | `docs/deviations.md` | read; all consistent with code |
| Lab log, results write-up, evidence table | `docs/experiment_log.md`, `docs/results.md`, `docs/evidence_table.md` | read |
| Config | `configs/train_config.json` | matches paper hyperparameters |
| Data generation | `scripts/prepare_data.py`, `scripts/build_condition_data.py`, `src/perturbations/`, `src/experiment/augment.py` | read; regenerable deterministically |
| Data | `data/splits/*.jsonl`, `data/perturbed/*.jsonl`, `MANIFEST.json` (SHA-256) | hashes verified for all 18 conditions |
| Training / evaluation | `scripts/train.py`, `src/experiment/modeling.py`, `scripts/run_all.py`, Colab notebook | read |
| Run records (55 runs + 1 excluded smoke run) | `results/runs/<run_id>/{run_config,train_log,metrics,predictions}` | all verified (see section 2) |
| Registry | `results/run_registry.csv` | 58 rows: 55 runs + 3 documented duplicate executions |
| Original analysis | `scripts/analyze.py`, `scripts/diag_distractor.py`, `results/analysis/` | reproduced byte-for-byte |
| Manuscripts | `paper/main.tex` (arXiv source, tracked) and an untracked 8-page IEEE draft | audited; superseded by `revised_manuscript/main.tex` and removed from the working tree (arXiv source kept in git history, commit `c96c067`) |
| Reference PDFs | `references/` (6 papers, git-ignored) | present locally |
| Not available | LoRA adapters (git-ignored, not on disk), passage/title ids, perturbed validation predictions, extra seeds, 1.5B runs, the local lab notebook `noise-diversity.docx` (not read) | see section 7 |

## 2. Reproduction of existing results

| check | result |
|---|---|
| `scripts/analyze.py` re-run on the stored records | `summary.md` and `summary.json` identical to `results/analysis/` |
| Accuracy recomputed from `predictions.json` vs `metrics.json` | identical for all 55 runs and 7 test versions |
| Training-data SHA-256 per run vs `MANIFEST.json` vs current files | all match |
| Train logs | every run has steps 1 to 500 in order |
| Primary result: pooled -0.14 pp, item-bootstrap CI [-0.59, +0.31] | **reproduced exactly** |
| Fold A -0.71 [-1.38, -0.05], fold B +0.42 [-0.24, +1.08], per-seed effects, seed SDs | reproduced exactly |
| 14 secondary contrasts, sweep, transfer matrix, distractor table | reproduced exactly |
| "15 of 4,000 training items truncated" | verified with the cached tokenizer (15 train, 1 val, 5 per test version; matches the run counters 53/54/36) |
| Per-family "changed" rates (80.0, 98.1, 77.5, 82.3, 100.0, 79.7%) | verified |
| 8.8M trainable params (1.75%), 500 steps, effective batch 16, T4 | verified from run records |
| Mean run time | 1,936 s training + 572 s evaluation (paper said 33 + 10 min; corrected to 32 + 10) |
| Determinism claim (three duplicate executions identical) | consistent with the registry; cannot be re-tested |

## 3. Findings, by severity

### Critical (would mislead a reader if left unchanged)

**C1. The primary interval ignores training-seed variance, so the "rules out a benefit larger than about 0.3 pp"
claim is too strong.** The pre-registered bootstrap resamples test items only. The seed SD of the pooled effect
(0.38 pp) is comparable to the item SE (0.23 pp). Combining them gives an approximate pooled interval of
**[-0.77, +0.48]**; a two-stage (seed x cluster) bootstrap gives **[-0.92, +0.62]**. The null conclusion stands,
but the bound on a possible benefit is about 0.5 to 0.6 pp, not 0.3 pp. The pre-registered rule (CI excludes 0
AND |effect| > seed SD) partly guards against this, and it is reported unchanged. Fixed in the revised manuscript.

**C2. The distractor finding was over-interpreted.** The original text says the 18-point drop happens "largely by
biasing it towards answering No" and is "rather than a general failure to understand perturbed text". The
reanalysis supports the shift (Yes rate -25.9 pp, CI [-28.7, -23.0]; 91% of lost items are Yes-to-No flips;
across 39 unexposed runs r = 0.81 between Yes-rate shift and accuracy drop) but also shows a **real loss of
discrimination**: AUC falls from 0.852 to 0.728 (0.747 within same-sentence groups). Label-free re-thresholding
recovers only 41% of the loss (52% with a per-sentence threshold). The revised paper describes both components
and avoids causal language.

### Major

**M1. Passage overlap was described inaccurately and it matters for fold A.** The paper says "136 of the 1,000
test passages also occur in the training split". In fact 136 test *questions* (121 of 969 distinct test passages)
have a verbatim train passage, and **173** do when near-duplicates (5-gram Jaccard >= 0.5, same Wikipedia
paragraph with small edits) are counted. Overlap items are 2 to 5 pp easier for every condition. Excluding them
moves fold A from -0.71 to -0.47 (verbatim) or -0.53 (near) pp, and fold A's item-level CI then **includes 0**;
the pooled estimate becomes 0.00 / -0.04. The claim "the overlap does not favour any condition" is therefore not
quite right: it is approximately balanced, but it shifts the fold A estimate by about 0.2 pp. The primary null is
unaffected. Fixed in the revised manuscript (corrected counts, sensitivity rows in Table II).

**M2. Clustered bootstrap.** Question-level resampling is formally inappropriate when questions share passages.
The passage-cluster bootstrap (966 clusters) gives practically identical intervals because 935 clusters are single
questions (pooled SE 0.232 vs 0.232 pp). Reported for completeness; no conclusion changes.

**M3. Exact p(Yes) = 0.5 ties are resolved to "No".** Trained runs have a median of 9 (range 0 to 31) items per
test version with p(Yes) exactly 0.5; no stored value lies in (0.49, 0.51) other than 0.5, consistent with fp16
logit ties. The rule `p > 0.5` sends all ties to "No", a small systematic push in the direction of the distractor
bias. Sensitivity: half credit or ties-to-Yes change primary estimates by at most 0.10 pp. Now disclosed.

**M4. Undisclosed software heterogeneity.** 17 runs used PyTorch 2.11.0 and 38 used 2.14.1 (same transformers
5.21.0 / peft 0.21.2). The version aligns mostly with seed index (seeds 0 and 1 early, seed 2 late), not with
condition, but the sweeps, EDA and NEFTune runs are all on 2.14.1 while two of three clean-baseline seeds are on
2.11.0. A small version effect cannot be excluded for those contrasts. Now disclosed as a threat.

**M5. Length vs venue.** The existing IEEE draft compiles to **8 pages**. SANKALP 2027 reportedly allows about 6
pages for the initial submission (to be confirmed, see checklist). The revised version is **6 pages including
references**, with appendices moved to the repository.

**M6. Novelty statement needs scoping.** "None of the text studies evaluates on held-out perturbation families" is
defensible for the six papers in the evidence table, but Karpukhin et al. (W-NUT 2019) train on a mixture of
synthetic noise and test on natural noise, Belinkov and Bisk (ICLR 2018) study cross-noise transfer, and vision
work (Mintun et al., NeurIPS 2021; Kang et al. 2019) already shows limited transfer to dissimilar corruptions.
The revised paper positions the contribution as a controlled, fixed-volume, family-held-out test for a small
instruction-tuned LM, not as a first-ever result.

**M7. "As few as 667 examples" understated the dose result.** Fold B Mixed-3 at 10% corruption contains 133
distractor examples and already reaches 77.6% distractor accuracy (clean-trained: 60.8%); with 333 examples it
matches clean-input accuracy. Descriptive only (three seeds per point).

### Minor

* Fold A secondary deficits (Mixed-3 vs clean -2.29, NEFTune -2.06, EDA -1.24 pp) were uncorrected in the
  protocol. They survive Holm adjustment over the 14 contrasts (adjusted p <= 0.044), the combined
  item-plus-seed interval, and the no-overlap subset. Retained, labelled as secondary.
* The fold B "seen" contrast (+4.51 pp) is correctly explained in the original as a design artefact; retained.
* The threshold-free score (probability on the gold answer) gives fold A -0.85 [-1.24, -0.46] (item CI only): the
  same direction as accuracy, slightly stronger. It does not support H1 in any setting.
* Five test items per version have truncated passages (identical truncation for all models); not material.
* The "meaningful" rule compares |effect| with the SD of single-seed effects, not the SE of the 3-seed mean. It is
  a conservative heuristic, not a test; the paper now says so implicitly by reporting the combined interval.
* The test set comes from the BoolQ *training* split; Qwen2.5 pretraining may have seen BoolQ. This can inflate
  absolute accuracy but applies equally to all conditions.
* Holdout-safe EDA uses different operations per fold (D2), so the EDA baseline is not identical across folds.
  Already documented.
* `docs/results.md` describes 136 "test passages"; corrected wording is in this audit and the revised paper. The
  historical document is left as written, with a pointer added in the experiment log.

## 4. Protocol: planned vs executed vs post hoc

| item | planned (protocol_v1) | executed | status |
|---|---|---|---|
| Runs | 19 conditions x 3 seeds = 57 | 18 x 3 = 54 + untuned | deviation D1 (logged before test results) |
| EDA baseline | generic edit augmentation | holdout-safe subset per fold | D2 |
| Synonym determinism | n/a | bug fixed before training | D3 |
| Evaluation | not specified in detail (exact match) | Yes/No logit comparison | D4 |
| Loss / truncation | n/a | answer-only loss; passage truncation | D5 |
| Hardware | n/a | Colab T4, fp16 autocast | D6 |
| 1.5B scale check | optional | not run | stated as limitation |
| Statistics | paired item bootstrap + seed-SD rule | as planned | reproduced |
| Cluster bootstrap, combined seed+item interval, two-stage bootstrap, overlap exclusion, tie rules, p(gold), Holm, per-family CIs, transfer CIs, distractor AUC / re-threshold / per-sentence / dose | not planned | **post hoc (2026-11-20)** | labelled as post hoc in paper and code |
| PyTorch version | not specified | 2.11.0 (17 runs) and 2.14.1 (38 runs) | undocumented until now |

## 5. Which contribution claims are supported

| proposed claim | verdict | evidence |
|---|---|---|
| 1. Diversity does not automatically improve generalisation to unseen families | **Supported, scoped** to this model/task/families | primary null in every setting; pooled benefit > ~0.5 pp ruled out |
| 2. Families have different transfer characteristics | **Supported, mostly via one family** | 5 families leave little headroom (diagonal < 1 pp); distractor +18.1 pp in-family, off-diagonal -2.6 to +1.4 |
| 3. Distractor noise causes answer-preference shifts | **Supported as an observed association, plus a discrimination loss** | Yes-rate shift, flip direction, r = 0.81; AUC drop; partial recovery by re-thresholding |
| 4. Fixed corruption volume gives a controlled test of diversity | **Supported (design property)** | all conditions 4,000 examples, same questions corrupted, 500 steps |
| 5. Family-held-out evaluation reveals weaknesses hidden by in-family evaluation | **Partly supported** | true for distractor (21.6 pp seen-vs-held-out gap); for the other five families the gap is < 3 pp in either direction. The paper does not compare with a random-split protocol, so "random-split" wording is avoided |

## 6. Statistical reanalysis summary (all post hoc)

| contrast | original (item CI) | cluster CI | no near-overlap (827 items) | approx. item+seed CI |
|---|---|---|---|---|
| Fold A primary | -0.71 [-1.38, -0.05] | [-1.39, -0.03] | -0.53 [-1.24, +0.18] | [-2.13, +0.72] |
| Fold B primary | +0.42 [-0.24, +1.08] | [-0.25, +1.06] | +0.46 [-0.30, +1.19] | [-0.43, +1.27] |
| Pooled primary | -0.14 [-0.59, +0.31] | [-0.60, +0.30] | -0.04 [-0.52, +0.44] | [-0.77, +0.48] |

Seed-level view: fold A per-seed -1.17, -1.51, +0.56 (t interval over seeds [-3.46, +2.04]); fold B +0.88, +0.47,
-0.08 ([-0.77, +1.61]). Leave-one-seed-out keeps the sign in each fold (fold A: -0.48, -0.31, -1.34; fold B: +0.19, +0.40,
+0.67 pp), so no single seed drives either fold's estimate. Per-family: no held-out family shows a
Mixed-3 advantage whose combined interval excludes 0. Full tables: `statistical_reanalysis/outputs/03_*.md`, `04_*.md`.

## 7. What cannot be established with the available artifacts

* Whether diversity helps at larger scale, on other tasks, or with stronger or natural perturbations.
* A precise estimate of training variance (three seeds only).
* Any mechanism for the distractor effect (adapters not available; only rounded p(Yes) is stored).
* Whether the in-family distractor gain generalises beyond the eight shared sentences (no held-out sentences).
* Whether the "No" shift comes from the missing punctuation boundary (only one construction exists).
* Calibration fitted on held-out data (no perturbed validation predictions).

## 8. Readiness assessment

The core experiment is careful (pre-registration, deviation log, matched volume, seed-matched design, full run
records, exact reproducibility). The main risks for acceptance are scope (one 0.5B model, one task, mostly mild
perturbations, asymmetric folds) and the fact that a null result needs a clear contribution statement. The revised
manuscript addresses the statistical and wording problems found here; it cannot address scope without new GPU
runs. See `submission_checklist.md` for remaining manual steps and the final assessment in `change_log.md`.
