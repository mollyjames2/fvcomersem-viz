#!/usr/bin/env bash
# Clear outputs from all Jupyter notebooks recursively.
# Requires: jupyter (nbconvert)

set -euo pipefail

# Find all .ipynb files (skip checkpoints) and clear outputs in-place
find . -type f -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -print0 \
| xargs -0 -I{} jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "{}"

echo " Cleared outputs from notebooks."

