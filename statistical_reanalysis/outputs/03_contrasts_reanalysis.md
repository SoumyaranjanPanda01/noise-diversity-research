# 03 Primary and secondary contrasts: post-hoc sensitivity (auto-generated)

Effects are Mixed-3 (50%) minus comparator, in percentage points (pp). Brackets: 95% CI.
`item/cluster CI` = test-item uncertainty only (bootstrap). `seed t-CI` = training-seed uncertainty only
(t interval, df = 2). `combined` = effect +- 1.96 sqrt(SE_item^2 + SD_seed^2/3), an approximation.

## Primary contrast under every setting

| setting | n items | fold A | fold B | pooled |
|---|---|---|---|---|
| S0 original (item bootstrap) | 1000 | -0.71 [-1.38, -0.05] | +0.42 [-0.24, +1.08] | -0.14 [-0.59, +0.31] |
| S1 cluster bootstrap | 1000 | -0.71 [-1.39, -0.03] | +0.42 [-0.25, +1.06] | -0.14 [-0.60, +0.30] |
| S2 cluster, exclude verbatim train passages | 864 | -0.47 [-1.17, +0.21] | +0.48 [-0.23, +1.18] | +0.00 [-0.48, +0.47] |
| S3 cluster, exclude verbatim+near-duplicate | 827 | -0.53 [-1.24, +0.18] | +0.46 [-0.30, +1.19] | -0.04 [-0.52, +0.44] |
| S4 cluster, ties = half credit | 1000 | -0.76 [-1.44, -0.09] | +0.42 [-0.24, +1.05] | -0.17 [-0.62, +0.27] |
| S5 cluster, ties = Yes | 1000 | -0.81 [-1.50, -0.12] | +0.42 [-0.25, +1.06] | -0.19 [-0.65, +0.26] |
| S6 cluster, p(gold) (threshold-free) | 1000 | -0.85 [-1.24, -0.46] | +0.33 [-0.09, +0.75] | -0.26 [-0.53, +0.01] |

## Primary contrast: separating item and seed uncertainty (S1, all items)

| fold | effect | cluster CI | per-seed effects | seed SD | seed t-CI | combined CI | seed x cluster bootstrap CI | protocol verdict |
|---|---|---|---|---|---|---|---|---|
| fold_a | -0.71 | [-1.39, -0.03] | -1.17, -1.51, +0.56 | 1.11 | [-3.46, +2.04] | [-2.13, +0.72] | [-2.18, +0.76] | CI excludes 0 but |effect| <= seed SD (not meaningful) |
| fold_b | +0.42 | [-0.25, +1.06] | +0.88, +0.47, -0.08 | 0.48 | [-0.77, +1.61] | [-0.43, +1.27] | [-0.62, +1.39] | null |
| pooled | -0.14 | [-0.60, +0.30] | -0.14, -0.52, +0.24 | 0.38 | [-1.09, +0.80] | [-0.77, +0.48] | [-0.92, +0.62] | null |

Item-bootstrap SE (S0) vs cluster-bootstrap SE (S1), pooled: 0.232 vs 0.232 pp.

## Leave-one-seed-out (S1)

* fold_a: without seed 0: -0.48, without seed 1: -0.31, without seed 2: -1.34
* fold_b: without seed 0: +0.19, without seed 1: +0.40, without seed 2: +0.67

## Per-seed effect with its own cluster CI (S1)

* fold_a: seed 0: -1.17 [-2.24, -0.08]; seed 1: -1.51 [-2.57, -0.47]; seed 2: +0.56 [-0.48, +1.59]
* fold_b: seed 0: +0.88 [-0.11, +1.86]; seed 1: +0.47 [-0.51, +1.46]; seed 2: -0.08 [-1.15, +1.01]

## Primary contrast by unseen family (S1)

| fold | family | effect | cluster CI | seed SD | combined CI |
|---|---|---|---|---|---|
| fold_a | shuffle | -1.04 | [-2.05, -0.06] | 1.12 | [-2.66, +0.57] |
| fold_a | distractor | -0.82 | [-1.87, +0.21] | 2.27 | [-3.60, +1.96] |
| fold_a | synonym | -0.26 | [-1.23, +0.68] | 0.83 | [-1.60, +1.09] |
| fold_b | typo | +0.63 | [-0.25, +1.47] | 0.72 | [-0.55, +1.82] |
| fold_b | formatting | +0.41 | [-0.46, +1.27] | 0.24 | [-0.50, +1.32] |
| fold_b | deletion | +0.22 | [-0.71, +1.14] | 0.53 | [-0.89, +1.33] |

## Mixed-3 vs each single-family model, unseen families (S1)

| fold | comparator | effect | cluster CI | combined CI |
|---|---|---|---|---|
| fold_a | single_typo | -0.68 | [-1.55, +0.19] | [-2.33, +0.98] |
| fold_a | single_formatting | -1.01 | [-1.77, -0.26] | [-2.84, +0.82] |
| fold_a | single_deletion | -0.43 | [-1.30, +0.43] | [-2.35, +1.49] |
| fold_b | single_shuffle | +0.74 | [-0.11, +1.60] | [-0.47, +1.96] |
| fold_b | single_distractor | +0.20 | [-0.54, +0.93] | [-0.80, +1.20] |
| fold_b | single_synonym | +0.32 | [-0.53, +1.19] | [-0.56, +1.21] |

## Secondary contrasts (S0 original vs S1 cluster; S1 adds combined CI and Holm-adjusted p)

| contrast | S0 effect [item CI] | S1 [cluster CI] | S3 disjoint-near [CI] | combined CI | seed SD | Holm p | protocol verdict (S1) |
|---|---|---|---|---|---|---|---|
| fold_a | clean baseline | unseen | -2.29 [-3.14, -1.48] | -2.29 [-3.12, -1.46] | -2.26 [-3.21, -1.35] | [-3.72, -0.85] | 1.03 | 0.000 | meaningful |
| fold_a | NEFTune | unseen | -2.06 [-2.92, -1.17] | -2.06 [-2.95, -1.17] | -2.19 [-3.15, -1.25] | [-2.98, -1.13] | 0.20 | 0.000 | meaningful |
| fold_a | EDA augmentation | unseen | -1.24 [-2.07, -0.40] | -1.24 [-2.10, -0.42] | -1.09 [-1.99, -0.19] | [-2.26, -0.23] | 0.50 | 0.044 | meaningful |
| fold_a | best single (formatting) | unseen | -1.01 [-1.78, -0.27] | -1.01 [-1.77, -0.26] | -0.67 [-1.49, +0.15] | [-2.84, +0.82] | 1.48 | 0.076 | CI excludes 0 but |effect| <= seed SD (not meaningful) |
| fold_a | mean single | seen | -0.32 [-1.00, +0.35] | -0.32 [-0.98, +0.36] | -0.32 [-1.07, +0.41] | [-1.64, +1.00] | 1.00 | 1.000 | null |
| fold_a | mean single | clean | -0.74 [-1.63, +0.12] | -0.74 [-1.66, +0.13] | -0.69 [-1.69, +0.28] | [-1.88, +0.39] | 0.62 | 0.830 | null |
| fold_a | clean baseline | clean | -0.57 [-1.70, +0.53] | -0.57 [-1.67, +0.56] | -0.56 [-1.81, +0.65] | [-1.79, +0.66] | 0.47 | 1.000 | null |
| fold_b | clean baseline | unseen | +0.37 [-0.43, +1.17] | +0.37 [-0.43, +1.14] | +0.36 [-0.49, +1.22] | [-0.64, +1.38] | 0.57 | 1.000 | null |
| fold_b | NEFTune | unseen | +0.06 [-0.77, +0.88] | +0.06 [-0.75, +0.85] | -0.05 [-0.94, +0.83] | [-1.14, +1.25] | 0.78 | 1.000 | null |
| fold_b | EDA augmentation | unseen | -0.32 [-1.11, +0.48] | -0.32 [-1.11, +0.47] | -0.40 [-1.26, +0.47] | [-1.35, +0.71] | 0.58 | 1.000 | null |
| fold_b | best single (distractor) | unseen | +0.20 [-0.54, +0.96] | +0.20 [-0.54, +0.93] | +0.13 [-0.70, +0.97] | [-0.80, +1.20] | 0.60 | 1.000 | null |
| fold_b | mean single | seen | +4.51 [+3.61, +5.40] | +4.51 [+3.60, +5.41] | +4.56 [+3.58, +5.57] | [+3.17, +5.85] | 0.88 | 0.000 | meaningful |
| fold_b | mean single | clean | +0.83 [-0.02, +1.67] | +0.83 [-0.01, +1.69] | +0.81 [-0.16, +1.74] | [-0.27, +1.93] | 0.62 | 0.464 | null |
| fold_b | clean baseline | clean | +0.70 [-0.37, +1.77] | +0.70 [-0.37, +1.78] | +0.73 [-0.44, +1.94] | [-1.04, +2.44] | 1.21 | 1.000 | null |

## Transfer matrix with cluster CIs (single-family model minus clean baseline, pp; S1)

| trained on | typo | formatting | deletion | shuffle | distractor | synonym | clean |
|---|---|---|---|---|---|---|---|
| typo | -0.7 [-2.0, +0.6] | -0.9 [-2.1, +0.4] | -1.3 [-2.4, -0.3] | -0.8 [-2.0, +0.4] | -2.1 [-3.5, -0.7] | -1.9 [-3.2, -0.6] | -0.4 [-1.5, +0.7] |
| formatting | +0.1 [-1.0, +1.3] | +0.3 [-1.0, +1.7] | +0.5 [-0.6, +1.5] | -0.3 [-1.4, +0.8] | -2.6 [-4.0, -1.4] | -0.9 [-2.1, +0.3] | +1.2 [+0.1, +2.3] |
| deletion | -0.2 [-1.4, +0.9] | -0.5 [-1.5, +0.6] | -0.5 [-1.7, +0.6] | -1.2 [-2.4, -0.0] | -2.6 [-4.1, -1.1] | -1.8 [-3.0, -0.6] | -0.2 [-1.3, +0.9] |
| shuffle | +0.2 [-0.8, +1.2] | -1.1 [-2.1, -0.1] | -0.2 [-1.2, +0.8] | -0.3 [-1.4, +0.8] | -1.4 [-2.7, +0.0] | -0.6 [-1.8, +0.5] | -0.3 [-1.3, +0.7] |
| distractor | -0.4 [-1.4, +0.6] | +0.3 [-0.6, +1.2] | +0.6 [-0.5, +1.7] | -1.3 [-2.3, -0.2] | +18.1 [+15.0, +21.2] | -0.1 [-1.3, +1.1] | -0.4 [-1.3, +0.6] |
| synonym | +0.0 [-1.2, +1.2] | -0.1 [-1.2, +0.9] | +0.3 [-0.9, +1.4] | -1.0 [-2.1, +0.1] | +1.4 [+0.2, +2.7] | +0.8 [-0.6, +2.1] | +0.3 [-0.8, +1.4] |

## Accuracy on test items with vs without a verbatim train passage (descriptive, %)

| condition | clean: overlap | clean: no overlap | 6 perturbed sets: overlap | 6 perturbed: no overlap |
|---|---|---|---|---|
| shared/clean | 81.9 | 78.3 | 78.0 | 73.5 |
| fold_a/mixed3_f0.50 | 81.1 | 77.8 | 75.7 | 72.1 |
| fold_b/mixed3_f0.50 | 82.1 | 79.1 | 80.5 | 76.9 |
| fold_a/single_typo | 83.1 | 77.7 | 77.0 | 72.1 |
| fold_a/single_formatting | 82.6 | 79.6 | 78.2 | 72.9 |
| fold_a/single_deletion | 82.1 | 78.0 | 76.5 | 72.4 |
| fold_b/single_shuffle | 81.9 | 77.9 | 76.8 | 73.0 |
| fold_b/single_distractor | 80.4 | 78.1 | 80.0 | 76.5 |
| fold_b/single_synonym | 82.6 | 78.5 | 78.6 | 73.6 |
