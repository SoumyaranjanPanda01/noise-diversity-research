# Noise Diversity vs. Corruption Volume

**Question.** At a fixed number of corrupted training examples (50%), does LoRA fine-tuning on a
*mixture* of three perturbation families (Mixed-3) generalise better to **unseen** perturbation
families than training on a single family?

Model: Qwen2.5-0.5B-Instruct + LoRA (r=16, α=32). Task: BoolQ (4,000 train / 500 val / 500 dev /
1,000 test). Six label-preserving perturbation families in a two-fold, family-held-out design:

| | train (seen) | evaluate (unseen) |
|---|---|---|
| Fold A | typo, formatting, deletion | shuffle, distractor, synonym |
| Fold B | shuffle, distractor, synonym | typo, formatting, deletion |

Full protocol: [`docs/protocol_v1.md`](docs/protocol_v1.md) · deviations & decisions:
[`docs/deviations.md`](docs/deviations.md) · lab log: [`docs/experiment_log.md`](docs/experiment_log.md) ·
related work: [`docs/evidence_table.md`](docs/evidence_table.md).

## Conditions (18 × 3 seeds = 54 training runs + 1 untuned eval)

| category | conditions |
|---|---|
| shared | `shared/clean` (0% noise), `shared/neftune` (clean + NEFTune α=5) |
| per fold | `single_<fam>` ×3 (50%), `mixed3_f0.50` (primary), sweeps `mixed3_f0.10/0.25/1.00`, `augment` (holdout-safe EDA, 50%) |

All conditions: 4,000 examples, 500 optimizer steps, identical hyperparameters (`configs/train_config.json`).

## Pipeline

```bash
pip install -r requirements.txt
python scripts/prepare_data.py            # splits + single/Mixed-3/test perturbations (deterministic)
python scripts/build_condition_data.py    # sweeps + EDA baseline + data/perturbed/MANIFEST.json (sha256)
python scripts/audit_data.py              # data quality gate
pytest -q                                 # perturbation + condition tests

python scripts/run_all.py --dry-run       # plan / status (55 runs)
python scripts/run_all.py                 # resumable; needs a CUDA GPU in practice
python scripts/analyze.py                 # stats -> results/analysis/summary.{md,json}, figures
```

A GPU is effectively required (a single run is ~14 h on a laptop CPU). Use
[`notebooks/run_experiments_gpu.ipynb`](notebooks/run_experiments_gpu.ipynb) on Colab
(free T4). It is resumable and keeps results on Google Drive.

One run: `python scripts/train.py --condition fold_a/mixed3_f0.50 --seed 0`.
Debug: add `--max-steps 3 --eval-limit 32 --tag smoke`. Tagged runs are excluded from the registry and the analysis.

## Records

| path | contents | tracked |
|---|---|---|
| `results/runs/<run_id>/run_config.json` | resolved config, data sha256, git commit, library versions, device | yes |
| `results/runs/<run_id>/train_log.jsonl` | per-step loss, grad-norm, lr, elapsed | yes |
| `results/runs/<run_id>/metrics.json` | val acc per epoch, test acc per eval set, timings | yes |
| `results/runs/<run_id>/predictions.json` | p(Yes) for every val/test item (paired stats) | yes |
| `results/run_registry.csv` | one row per completed run | yes |
| `results/analysis/` | primary/secondary contrasts, sweep, transfer matrix, figures | yes |
| `results/checkpoints/` | LoRA adapters | no (large, reproducible) |

## Analysis

The primary effect is acc(Mixed-3) − mean acc(single families) on the fold's unseen families,
averaged over seeds. The CI is a 10,000-resample paired bootstrap over the 1,000 test questions, applied
jointly across families, seeds and conditions. An effect counts as meaningful only if the 95% CI excludes 0
**and** it exceeds the cross-seed SD. A null result is a valid outcome.
