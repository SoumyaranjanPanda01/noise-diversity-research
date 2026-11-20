# Revised manuscript (IEEE conference format)

* `main.tex`: revised 6-page IEEE version (IEEEtran `conference`), written 2026-11-20 from the audit in
  `../research_audit.md`. It replaces the earlier drafts (the arXiv source `paper/main.tex` remains in git history,
  commit `c96c067`).
* `figures/`: the three figure PDFs, written by `../statistical_reanalysis/05_figures.py`.
* `main.pdf`: compiled output (6 pages, all fonts embedded).

Build: `sh revised_manuscript/build.sh` from the repository root (regenerates the analysis and figures,
compiles with pdflatex or tectonic). On Overleaf: upload `main.tex` and `figures/`, then compile with pdfLaTeX.

Main differences from the earlier IEEE draft are listed in `../change_log.md`.
