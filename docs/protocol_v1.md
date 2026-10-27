# Protocol Sketch v1: Noise Diversity vs. Corruption Volume

## 1. Core Research Question & Primary Comparison
* **Primary Question:** At a fixed number of corrupted examples (50% corruption volume), does training on a mixture of distinct perturbation families (Mixed-3) outperform single-family training when evaluated on unseen perturbation families[cite: 37, 40, 41]?
* **Primary Comparison:** Mixed-3 accuracy minus the mean accuracy of the 3 single-family models on unseen noise families[cite: 41, 42].
* **Statistical Criterion:** Evaluated using a paired bootstrap 95% confidence interval over test items across random seeds[cite: 42]. An effect is meaningful only if the CI excludes zero and exceeds cross-seed variation[cite: 42]. Null results are considered fully valid outcomes[cite: 37, 42].

---

## 2. Experimental Design & Folds
A two-fold cross-validation setup over 6 label-preserving perturbation families, ensuring complete family-level holdout[cite: 40, 41]:

* **Fold A:**
  * **Train (Seen):** `typo`, `formatting`, `deletion`[cite: 41]
  * **Evaluation (Unseen):** `shuffle`, `distractor`, `synonym`[cite: 41]
* **Fold B (Exact Reverse):**
  * **Train (Seen):** `shuffle`, `distractor`, `synonym`[cite: 41]
  * **Evaluation (Unseen):** `typo`, `formatting`, `deletion`[cite: 41]
* **Leakage Safeguards:** Test perturbations are generated strictly from the test partition using independent random seeds; test inputs never appear in training[cite: 6, 7, 41].

---

## 3. Model Architecture & Hyperparameters
* **Base Model:** `Qwen2.5-0.5B-Instruct` (with `Qwen2.5-1.5B-Instruct` reserved strictly as an optional scale check)[cite: 40].
* **Fine-Tuning Method:** LoRA applied to attention and MLP projection layers[cite: 41].
  * Rank ($r$): 16[cite: 41]
  * Alpha ($\alpha$): 32[cite: 41]
  * Dropout: 0.05[cite: 41]
* **Optimizer & Schedule:** AdamW, learning rate $2 \times 10^{-4}$, cosine decay with short warmup[cite: 41].
* **Training Budget:** 2 epochs, effective batch size 16, max sequence length 512 tokens[cite: 41].
* **Tuning Guardrail:** Hyperparameter modifications permitted on the development set only; zero tuning on test data[cite: 41].

---

## 4. Dataset Splits
Standardized sample partition with short, exactly-scorable target answers (e.g., BoolQ)[cite: 40]:
* **Train:** 4,000 examples[cite: 40]
* **Validation:** 500 examples[cite: 40]
* **Development:** 500 examples[cite: 40]
* **Test:** 1,000 examples (evaluated once post-protocol freeze)[cite: 40]

---

## 5. Condition Matrix & Run Accounting (57 Runs)
All conditions use matched sample counts and optimization steps[cite: 38, 41].

| Condition Category | Descriptions / Variants | Corruption Fraction | Count per Fold | Total Conditions |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Baseline** | Uncorrupted instruction tuning (shared across folds) | 0.0 | — | 1 |
| **Single-Family** | 1 family per condition (3 individual families per fold) | 0.5 (2,000 noisy / 2,000 clean) | 3 | 6 |
| **Mixed-3 (Primary)** | Equal split across 3 fold families | 0.5 (~667 per family = 2,000 noisy) | 1 | 2 |
| **Mixed-3 Sweeps** | Volume scaling of Mixed-3 | 0.1, 0.25, 1.0 | 3 | 6 |
| **Embedding Baseline**| NEFTune representation noise | N/A (latent embedding noise) | 1 | 2 |
| **Augmentation Baseline**| Generic edit-based data augmentation | Matched corrupted size | 1 | 2 |

* **Condition Total:** $1\text{ (Clean)} + 2 \times 9\text{ (Fold A & B specific conditions)} = 19\text{ unique conditions}$[cite: 41].
* **Total Runs:** $19\text{ conditions} \times 3\text{ random seeds} = \mathbf{57\text{ runs}}$[cite: 42].