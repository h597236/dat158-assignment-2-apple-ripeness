#!/usr/bin/env bash
# Lastar ned Fuji Ripeness & Size Dataset frå Kaggle.
# Krev Kaggle API-nøkkel: https://www.kaggle.com/settings -> "Create New Token",
# legg kaggle.json i ~/.kaggle/ (Linux/Mac) eller %USERPROFILE%\.kaggle\ (Windows).
#
# Kjelde: Zhu et al. (2025), "Foundation Model-Based Apple Ripeness and Size
# Estimation for Selective Harvesting", arXiv:2502.01850
set -euo pipefail

DEST="data/raw"
mkdir -p "$DEST"
kaggle datasets download -d zhukeyi1/fuji-ripeness-and-size-dataset -p "$DEST" --unzip

echo
echo "Ferdig. Innhald i $DEST:"
find "$DEST" -maxdepth 3 -type d | head -30
echo
echo "Annotasjonsfiler funne:"
find "$DEST" \( -name "*.json" -o -name "*.xml" \) | head -20
echo "(+ $(find "$DEST" -name '*.txt' | wc -l) .txt-filer)"
