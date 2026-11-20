#!/usr/bin/env sh
# Rebuild the revised manuscript from the repository root:  sh revised_manuscript/build.sh
# 1. regenerate the reanalysis outputs and figures (CPU only, about 2 minutes)
# 2. compile with pdflatex (TeX Live / MiKTeX / Overleaf) or tectonic, whichever is available
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
(cd statistical_reanalysis && $PY 01_verify_records.py && $PY 02_passage_overlap.py && \
 $PY 03_contrasts_reanalysis.py && $PY 04_distractor_bias.py && $PY 05_figures.py)
cd revised_manuscript
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
else
  echo "No TeX engine found: upload revised_manuscript/ (main.tex + figures/) to Overleaf." >&2
  exit 1
fi
