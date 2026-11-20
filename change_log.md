# Change log (2026-11-20)

All changes made during the pre-submission audit, with reasons. No model was trained or evaluated. Files under
`results/`, `data/`, `src/`, `scripts/`, `configs/`, `notebooks/` and `tests/` were not modified (verified by
SHA-256 for `results/` and by `git status` for the rest).

## Added

| path | what | why |
|---|---|---|
| `statistical_reanalysis/` (5 scripts, `common.py`, `run_all.sh`, README) | post-hoc reanalysis on CPU | audit the statistics without new GPU runs |
| `statistical_reanalysis/outputs/` | integrity report, SHA-256 of original records, overlap flags per test item, contrasts under 7 settings, distractor analysis, per-run CSV | reproducible evidence for every new number in the paper |
| `revised_manuscript/main.tex`, `main.pdf`, `figures/`, `build.sh`, README | 6-page IEEE version | target venue limit; corrected claims |
| `research_audit.md` | audit of methodology, implementation, results and limitations | deliverable |
| `literature_review.md` | closest prior work, novelty, verification status of each reference | deliverable |
| `submission_checklist.md` | venue facts to confirm and remaining manual steps | deliverable |
| README section and experiment-log entry | pointers to the post-hoc work | keep the historical record explicit |

## Removed (working tree only; tracked files remain in git history)

| path | reason |
|---|---|
| `bu.pdf`, `openreview.pdf` (untracked) | not related to the IEEE submission (removed on request) |
| `paper/main_acl.tex`, `paper/acl.sty` (untracked) | ACL-format draft; IEEE is the target |
| `paper/main_ieee.tex` (untracked) | 8-page IEEE draft, superseded by `revised_manuscript/main.tex` |
| `paper/main.tex` (tracked, `git rm`) | arXiv-format draft, superseded; recoverable from commit `c96c067` |
| `paper/*.pdf`, `arxiv/`, `docs/paper_explained.pdf`, `scripts/make_paper_guide.py` (git-ignored) | build outputs and local study material, not needed |
| caches (`__pycache__/`, `.pytest_cache/`) | clutter |
| `.gitignore` entries for removed items | replaced with entries for `revised_manuscript/` build outputs |

Kept although not committed: `noise-diversity.docx` (lab notebook), `references/` (downloaded papers), `venv/`.
These are local-only and git-ignored.

## Manuscript changes (vs the earlier 8-page IEEE draft)

| change | reason (audit item) |
|---|---|
| Pooled interval reported with seed variance, [-0.77, +0.48]; "rules out > 0.3 pp" replaced by "about half a point" | C1: item-only interval ignored training variance |
| Distractor result rewritten: answer shift **and** discrimination loss (AUC 0.852 to 0.728); re-thresholding recovers 41 to 52%; no "rather than a failure of comprehension" | C2: over-interpretation |
| Overlap corrected to 136 questions / 121 of 969 passages / 173 with near-duplicates; overlap-excluded results added | M1 |
| Passage-cluster bootstrap added (Table II) | M2 |
| fp16 ties disclosed with sensitivity | M3 |
| PyTorch version mix disclosed under threats to validity | M4 |
| Appendices removed (full table, hyperparameter table, deviation list now in the repository); related work compressed; 8 to 6 pages | M5 |
| Contribution scoped; Karpukhin et al., Mintun et al., Gulrajani and Lopez-Paz, Jia and Liang, Zhao et al. and others added | M6 |
| "As few as 667" replaced by the dose-response (133 examples suffice) | M7 |
| New sections: Research Questions; Threats to Validity and Limitations; sensitivity subsection | requested structure, kept within 6 pages |
| Fold A secondary contrasts kept, now with Holm-adjusted p and combined intervals | robustness of a reported finding |
| Mean run time 33 to 32 min | recomputed from records (1,936 s) |
| Volume sweep reported in text only (figure dropped for space) | page limit |
| New figures: primary effect with all uncertainty views; transfer matrix with CI markers; distractor answer-rate and dose-response | generated from records by `05_figures.py` |
| Bibliography: 16 to 31 entries, IEEE numeric style, citation order | literature review |

## Code fix made during the audit

* `statistical_reanalysis/common.py`: the near-duplicate search first used `Counter.most_common(5)`, whose
  tie order depends on `PYTHONHASHSEED`, so low Jaccard values varied between runs (flags and all reported
  numbers did not). It now takes the exact maximum over all candidates; outputs are byte-identical across hash seeds.

## Final assessment

**Ready to submit after the manual steps in `submission_checklist.md`**, chiefly: confirm the page limit,
blind-review policy and deadline (30 November 2026) on the official SANKALP site; fix the author block if the
review is double-blind; verify three page ranges; push the new folders to GitHub so the availability statement
holds.

Remaining risks:
* **Scope** is the main weakness reviewers will raise: one 0.5B model, one task, synthetic question-only
  perturbations, five of six families with little headroom, asymmetric folds. None of this can be fixed without
  new GPU runs, so the paper states it plainly.
* A null result needs reviewers who value controlled negative evidence; the contribution statement and the
  distractor anatomy are what make the paper more than "nothing happened".
* Post-hoc analyses are clearly labelled, but the distractor section remains exploratory with three seeds.
* Venue facts could not be read from the official site (expired certificate); treat them as unconfirmed.
