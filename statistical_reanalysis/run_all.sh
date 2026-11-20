#!/usr/bin/env sh
# Runs the whole post-hoc reanalysis on CPU. Usage: sh statistical_reanalysis/run_all.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY 01_verify_records.py
$PY 02_passage_overlap.py
$PY 03_contrasts_reanalysis.py
$PY 04_distractor_bias.py
$PY 05_figures.py
