# 04 Distractor answer-preference shift (exploratory; auto-generated)

Gold Yes rate: 63.4%. Values: mean ± SD over seeds (untuned: 1 run). CIs: passage-cluster bootstrap over test items, seed-averaged.

## Yes rate, accuracy and discrimination, clean vs distractor

| condition | Yes% clean | Yes% distractor | Yes-rate shift [CI] | acc clean | acc distractor | acc change [CI] | AUC clean | AUC distractor | acc after label-free re-threshold |
|---|---|---|---|---|---|---|---|---|---|
| untuned | 37.1 ± 0.0 | 0.8 ± 0.0 | -36.3 [-39.4, -33.3] | 60.5 ± 0.0 | 37.4 ± 0.0 | -23.1 [-26.6, -19.5] | 0.719 ± 0.000 | 0.564 ± 0.000 | 51.6 ± 0.0 |
| shared/clean | 64.7 ± 0.8 | 38.8 ± 6.6 | -25.9 [-28.7, -23.0] | 78.8 ± 0.7 | 60.8 ± 3.2 | -18.0 [-21.1, -14.9] | 0.852 ± 0.006 | 0.728 ± 0.010 | 68.1 ± 1.6 |
| shared/neftune | 65.9 ± 3.4 | 35.0 ± 7.0 | -30.9 [-33.6, -28.1] | 79.7 ± 0.3 | 60.5 ± 3.4 | -19.2 [-22.4, -16.1] | 0.856 ± 0.008 | 0.752 ± 0.019 | 70.8 ± 0.7 |
| fold_a/single_typo | 64.6 ± 3.1 | 32.0 ± 2.1 | -32.6 [-35.4, -29.8] | 78.4 ± 1.0 | 58.7 ± 2.2 | -19.7 [-23.0, -16.5] | 0.850 ± 0.007 | 0.740 ± 0.010 | 69.7 ± 0.3 |
| fold_a/single_formatting | 65.4 ± 1.8 | 32.7 ± 12.2 | -32.7 [-35.4, -30.0] | 80.0 ± 0.6 | 58.2 ± 5.3 | -21.8 [-24.9, -18.7] | 0.855 ± 0.001 | 0.740 ± 0.014 | 69.9 ± 0.9 |
| fold_a/single_deletion | 65.1 ± 2.9 | 30.5 ± 1.5 | -34.6 [-37.3, -31.9] | 78.6 ± 0.9 | 58.2 ± 1.4 | -20.3 [-23.5, -17.2] | 0.847 ± 0.007 | 0.745 ± 0.010 | 70.4 ± 1.7 |
| fold_a/mixed3_f0.10 | 65.7 ± 2.9 | 40.0 ± 7.6 | -25.7 [-28.6, -22.8] | 78.9 ± 0.8 | 60.1 ± 2.6 | -18.8 [-21.8, -15.7] | 0.848 ± 0.012 | 0.722 ± 0.011 | 67.8 ± 1.4 |
| fold_a/mixed3_f0.25 | 64.9 ± 1.8 | 37.4 ± 10.4 | -27.5 [-30.3, -24.8] | 78.9 ± 1.0 | 60.9 ± 3.6 | -18.0 [-21.1, -14.9] | 0.853 ± 0.005 | 0.740 ± 0.004 | 69.6 ± 0.7 |
| fold_a/mixed3_f0.50 | 64.4 ± 2.2 | 31.1 ± 6.7 | -33.4 [-36.1, -30.5] | 78.2 ± 0.5 | 57.5 ± 3.2 | -20.7 [-23.9, -17.4] | 0.846 ± 0.004 | 0.724 ± 0.014 | 67.8 ± 1.7 |
| fold_a/mixed3_f1.00 | 64.0 ± 2.7 | 31.3 ± 2.8 | -32.7 [-35.5, -29.9] | 78.0 ± 0.6 | 57.2 ± 1.2 | -20.8 [-23.9, -17.6] | 0.839 ± 0.005 | 0.709 ± 0.024 | 66.9 ± 2.2 |
| fold_a/augment | 64.7 ± 1.9 | 31.0 ± 6.0 | -33.7 [-36.3, -31.0] | 79.4 ± 0.8 | 59.0 ± 2.9 | -20.4 [-23.5, -17.3] | 0.855 ± 0.003 | 0.754 ± 0.016 | 70.9 ± 1.3 |
| fold_b/single_shuffle | 64.9 ± 2.1 | 32.8 ± 2.6 | -32.0 [-34.7, -29.3] | 78.5 ± 0.4 | 59.4 ± 1.4 | -19.0 [-22.1, -15.9] | 0.846 ± 0.003 | 0.744 ± 0.013 | 69.7 ± 1.5 |
| fold_b/single_synonym | 65.6 ± 1.4 | 36.7 ± 4.2 | -28.9 [-31.5, -26.2] | 79.1 ± 0.4 | 62.2 ± 1.3 | -16.9 [-19.9, -13.9] | 0.848 ± 0.003 | 0.756 ± 0.014 | 70.8 ± 1.7 |
| fold_b/augment | 66.1 ± 2.1 | 36.9 ± 7.7 | -29.2 [-31.9, -26.5] | 79.7 ± 0.7 | 61.8 ± 4.1 | -17.8 [-20.8, -14.8] | 0.853 ± 0.004 | 0.748 ± 0.021 | 70.4 ± 1.4 |
| fold_b/mixed3_f0.10 | 64.7 ± 1.6 | 62.1 ± 3.0 | -2.6 [-3.9, -1.3] | 78.9 ± 0.8 | 77.6 ± 0.3 | -1.3 [-2.6, -0.0] | 0.852 ± 0.006 | 0.840 ± 0.006 | 77.9 ± 0.5 |
| fold_b/mixed3_f0.25 | 65.1 ± 1.2 | 65.5 ± 2.7 | +0.4 [-0.5, +1.3] | 79.0 ± 0.5 | 78.9 ± 0.7 | -0.1 [-1.0, +0.8] | 0.851 ± 0.002 | 0.845 ± 0.005 | 78.7 ± 0.5 |
| fold_b/mixed3_f0.50 | 64.9 ± 1.7 | 65.1 ± 1.7 | +0.2 [-0.8, +1.2] | 79.5 ± 0.7 | 79.1 ± 0.8 | -0.4 [-1.4, +0.5] | 0.853 ± 0.004 | 0.850 ± 0.003 | 78.9 ± 0.7 |
| fold_b/mixed3_f1.00 | 65.3 ± 2.1 | 64.4 ± 1.8 | -0.9 [-1.7, -0.1] | 78.9 ± 0.5 | 78.5 ± 0.5 | -0.4 [-1.2, +0.4] | 0.844 ± 0.004 | 0.844 ± 0.007 | 78.4 ± 0.4 |
| fold_b/single_distractor | 65.0 ± 1.1 | 65.3 ± 1.3 | +0.3 [-0.4, +1.0] | 78.4 ± 0.3 | 78.9 ± 0.8 | +0.5 [-0.3, +1.2] | 0.847 ± 0.003 | 0.848 ± 0.003 | 78.6 ± 0.5 |

## Per-seed shifts (pp): Yes-rate shift / accuracy change

* untuned: -36.3 / -23.1
* shared/clean: -18.9 / -14.9; -29.5 / -22.1; -29.2 / -17.0
* shared/neftune: -32.2 / -19.0; -35.2 / -22.8; -25.3 / -15.7
* fold_a/single_typo: -35.2 / -21.2; -30.7 / -21.3; -31.9 / -16.7
* fold_a/single_formatting: -20.5 / -15.5; -40.1 / -26.9; -37.6 / -23.0
* fold_a/single_deletion: -36.1 / -22.1; -29.8 / -18.2; -37.9 / -20.7
* fold_a/mixed3_f0.10: -22.4 / -17.0; -31.6 / -21.8; -23.1 / -17.5
* fold_a/mixed3_f0.25: -16.0 / -15.0; -31.5 / -20.5; -35.0 / -18.4
* fold_a/mixed3_f0.50: -28.1 / -18.7; -38.2 / -24.4; -33.8 / -19.0
* fold_a/mixed3_f1.00: -32.5 / -19.3; -29.9 / -22.1; -35.7 / -20.9
* fold_a/augment: -34.3 / -22.5; -37.6 / -22.2; -29.1 / -16.5
* fold_b/single_shuffle: -34.2 / -19.0; -32.0 / -20.4; -29.9 / -17.7
* fold_b/single_synonym: -30.6 / -17.8; -30.8 / -17.8; -25.2 / -15.0
* fold_b/augment: -38.4 / -22.4; -26.1 / -18.1; -23.2 / -13.0
* fold_b/mixed3_f0.10: -3.2 / -0.6; -3.7 / -2.3; -0.8 / -1.0
* fold_b/mixed3_f0.25: +0.3 / +0.1; -1.1 / -0.9; +1.9 / +0.5
* fold_b/mixed3_f0.50: -0.3 / -0.5; +0.1 / -0.3; +0.7 / -0.5
* fold_b/mixed3_f1.00: -0.8 / +0.2; -0.5 / -0.1; -1.5 / -1.3
* fold_b/single_distractor: +0.6 / +1.0; +0.2 / +0.4; +0.0 / +0.0

## Where the lost items go (mean over seeds)

| condition | correct->wrong via Yes->No | correct->wrong via No->Yes | wrong->correct | mean log-odds shift |
|---|---|---|---|---|
| untuned | 298.0 | 0.0 | 67.0 | -1.42 ± 0.00 |
| shared/clean | 241.7 | 24.7 | 86.3 | -1.58 ± 0.26 |
| shared/neftune | 263.3 | 13.7 | 85.3 | -1.70 ± 0.22 |
| fold_a/single_typo | 273.0 | 9.7 | 85.3 | -2.09 ± 0.15 |
| fold_a/single_formatting | 284.3 | 14.0 | 80.3 | -2.05 ± 0.47 |
| fold_a/single_deletion | 282.0 | 6.3 | 85.0 | -2.03 ± 0.35 |
| fold_a/mixed3_f0.10 | 247.3 | 27.7 | 87.3 | -1.53 ± 0.16 |
| fold_a/mixed3_f0.25 | 244.0 | 21.3 | 85.7 | -1.80 ± 0.49 |
| fold_a/mixed3_f0.50 | 284.3 | 10.7 | 88.0 | -2.02 ± 0.20 |
| fold_a/mixed3_f1.00 | 281.7 | 16.0 | 90.0 | -1.84 ± 0.35 |
| fold_a/augment | 278.7 | 5.3 | 80.0 | -2.04 ± 0.21 |
| fold_b/single_shuffle | 263.7 | 6.3 | 79.7 | -2.03 ± 0.32 |
| fold_b/single_synonym | 240.0 | 10.3 | 81.7 | -1.90 ± 0.21 |
| fold_b/augment | 250.3 | 13.7 | 85.7 | -1.86 ± 0.49 |
| fold_b/mixed3_f0.10 | 36.0 | 11.3 | 34.3 | -0.48 ± 0.12 |
| fold_b/mixed3_f0.25 | 10.7 | 13.3 | 23.0 | -0.13 ± 0.10 |
| fold_b/mixed3_f0.50 | 14.3 | 14.3 | 24.3 | -0.08 ± 0.13 |
| fold_b/mixed3_f1.00 | 13.3 | 6.7 | 16.0 | -0.23 ± 0.02 |
| fold_b/single_distractor | 6.0 | 6.7 | 17.3 | -0.02 ± 0.04 |

## By appended sentence (Yes-rate shift / accuracy change, pp, mean over seeds)

| sentence | n | untuned | clean | A Mixed-3 | B Mixed-3 |
|---|---|---|---|---|---|
| Photosynthesis is the process used by plants to ... | 137 | -36.5 / -27.7 | +6.1 / -2.2 | -2.7 / -4.1 | -1.2 / -2.2 |
| Mercury is the smallest planet in our solar syst... | 137 | -41.6 / -32.8 | -52.6 / -37.0 | -57.4 / -39.4 | -1.0 / +0.5 |
| Paris hosted the Summer Olympic Games in 2024.... | 111 | -41.4 / -19.8 | -42.6 / -25.8 | -50.2 / -26.1 | -1.8 / -0.6 |
| The Pacific Ocean is the largest and deepest oce... | 123 | -30.1 / -15.4 | +0.3 / -5.1 | -16.5 / -8.4 | +3.5 / +0.3 |
| The printing press was developed by Johannes Gut... | 96 | -27.1 / -14.6 | -9.4 / -9.4 | -19.1 / -14.2 | +4.9 / +1.4 |
| Mount Everest is the highest mountain peak above... | 145 | -33.8 / -25.5 | -18.9 / -15.6 | -22.3 / -16.8 | -0.5 / -1.4 |
| Jupiter has the shortest day of all the planets ... | 118 | -37.3 / -20.3 | -52.3 / -26.8 | -53.4 / -28.0 | -0.0 / +0.6 |
| Honey never completely spoils if kept in an airt... | 133 | -40.6 / -24.1 | -37.6 / -21.1 | -46.4 / -27.8 | -1.3 / -1.3 |

## Within-sentence discrimination and per-sentence label-free re-thresholding

AUC is computed inside each of the 8 sentence groups and averaged (weighted by group size). If within-group AUC
on distractor inputs stays close to clean, the pooled AUC loss comes mostly from sentence-specific shifts.

| condition | within AUC clean | within AUC distractor | acc distractor | acc, global re-threshold | acc, per-sentence re-threshold |
|---|---|---|---|---|---|
| untuned | 0.718 | 0.571 | 37.4 ± 0.0 | 51.6 ± 0.0 | 51.4 ± 0.0 |
| shared/clean | 0.852 | 0.747 | 60.8 ± 3.2 | 68.1 ± 1.6 | 70.1 ± 1.3 |
| shared/neftune | 0.856 | 0.769 | 60.5 ± 3.4 | 70.8 ± 0.7 | 72.4 ± 0.8 |
| fold_a/single_typo | 0.849 | 0.752 | 58.7 ± 2.2 | 69.7 ± 0.3 | 70.5 ± 0.7 |
| fold_a/single_formatting | 0.856 | 0.758 | 58.2 ± 5.3 | 69.9 ± 0.9 | 71.8 ± 0.4 |
| fold_a/single_deletion | 0.846 | 0.758 | 58.2 ± 1.4 | 70.4 ± 1.7 | 70.9 ± 1.3 |
| fold_a/mixed3_f0.10 | 0.847 | 0.737 | 60.1 ± 2.6 | 67.8 ± 1.4 | 69.7 ± 1.1 |
| fold_a/mixed3_f0.25 | 0.853 | 0.756 | 60.9 ± 3.6 | 69.6 ± 0.7 | 70.9 ± 0.8 |
| fold_a/mixed3_f0.50 | 0.846 | 0.743 | 57.5 ± 3.2 | 67.8 ± 1.7 | 69.9 ± 1.0 |
| fold_a/mixed3_f1.00 | 0.838 | 0.725 | 57.2 ± 1.2 | 66.9 ± 2.2 | 68.5 ± 2.1 |
| fold_a/augment | 0.855 | 0.766 | 59.0 ± 2.9 | 70.9 ± 1.3 | 71.8 ± 0.7 |
| fold_b/single_shuffle | 0.845 | 0.758 | 59.4 ± 1.4 | 69.7 ± 1.5 | 71.4 ± 1.6 |
| fold_b/single_synonym | 0.848 | 0.770 | 62.2 ± 1.3 | 70.8 ± 1.7 | 72.1 ± 1.1 |
| fold_b/augment | 0.853 | 0.769 | 61.8 ± 4.1 | 70.4 ± 1.4 | 72.3 ± 1.2 |
| fold_b/mixed3_f0.10 | 0.851 | 0.838 | 77.6 ± 0.3 | 77.9 ± 0.5 | 78.0 ± 0.7 |
| fold_b/mixed3_f0.25 | 0.851 | 0.845 | 78.9 ± 0.7 | 78.7 ± 0.5 | 78.9 ± 0.7 |
| fold_b/mixed3_f0.50 | 0.854 | 0.850 | 79.1 ± 0.8 | 78.9 ± 0.7 | 79.0 ± 0.7 |
| fold_b/mixed3_f1.00 | 0.844 | 0.844 | 78.5 ± 0.5 | 78.4 ± 0.4 | 78.6 ± 0.7 |
| fold_b/single_distractor | 0.847 | 0.849 | 78.9 ± 0.8 | 78.6 ± 0.5 | 78.6 ± 0.7 |

## Dose-response: distractor training examples vs distractor-test accuracy

| condition | distractor train examples | acc distractor | Yes% distractor |
|---|---|---|---|
| shared/clean | 0 | 60.8 ± 3.2 | 38.8 ± 6.6 |
| fold_b/augment | 0 | 61.8 ± 4.1 | 36.9 ± 7.7 |
| fold_b/single_shuffle | 0 | 59.4 ± 1.4 | 32.8 ± 2.6 |
| fold_b/single_synonym | 0 | 62.2 ± 1.3 | 36.7 ± 4.2 |
| fold_b/mixed3_f0.10 | 133 | 77.6 ± 0.3 | 62.1 ± 3.0 |
| fold_b/mixed3_f0.25 | 333 | 78.9 ± 0.7 | 65.5 ± 2.7 |
| fold_b/mixed3_f0.50 | 667 | 79.1 ± 0.8 | 65.1 ± 1.7 |
| fold_b/mixed3_f1.00 | 1333 | 78.5 ± 0.5 | 64.4 ± 1.8 |
| fold_b/single_distractor | 2000 | 78.9 ± 0.8 | 65.3 ± 1.3 |

Across the 39 trained runs with no distractor training data, the per-run Yes-rate shift and the per-run accuracy change correlate with Pearson r = 0.81 (Spearman 0.82). This is an association across runs, not a causal estimate.
