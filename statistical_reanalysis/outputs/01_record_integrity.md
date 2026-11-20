# 01 Record integrity and reproduction (auto-generated)

* Runs checked (excluding the tagged smoke run): **55**
* Integrity issues found: **0**
* `scripts/analyze.py` re-run: summary.md identical = **True**, summary.json identical = **True**
* Registry rows duplicated by Colab resumes: {'fold_b__mixed3_f0.25__s0': 2, 'fold_b__mixed3_f0.25__s1': 2, 'fold_b__mixed3_f1.00__s1': 2}
* Software environments (torch / transformers / peft / GPU): 2.14.1+cu130 / 5.21.0 / 0.21.2 / Tesla T4 = 38 runs; 2.11.0+cu130 / 5.21.0 / 0.21.2 / Tesla T4 = 17 runs
* Exact p(Yes) = 0.5 values per run and eval set (trained runs): min 0, max 31, mean 9.9 of 1,000. Untuned: {'clean': 30, 'typo': 22, 'formatting': 17, 'deletion': 31, 'shuffle': 31, 'distractor': 1, 'synonym': 10}.
  No predicted value lies in (0.49, 0.51) other than exactly 0.5, consistent with fp16 logit ties.
  The original rule p > 0.5 resolves every tie to "No". Sensitivity: see 03.
* SHA-256 of every file under results/: `outputs/original_records_sha256.txt` (287 files).
