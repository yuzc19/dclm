#!/bin/bash

set -euo pipefail

# Default parameters
TEXTFILES_DIR="/tmp/data/fasttext_0.1/3.6B_data"
SCORES_DIR="/tmp/data/fasttext_0.1/3.6B_data/10000-data_influence_model-flan-prediction"
OUTPUT_DIR="/tmp/data/fasttext_0.1/3.6B_data/10000-data_influence_model-flan-selection_with_replacement"
SHARD_NUM=8
RATIO=1
TEMP=0.5
SEED=1234

# Run selection with replacement
python mates/tokenization/select_data_replacement.py \
    --textfiles_dir "$TEXTFILES_DIR" \
    --scores_dir "$SCORES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --shard_num "$SHARD_NUM" \
    --ratio "$RATIO" \
    --temp "$TEMP" \
    --seed "$SEED"

echo "Selection with replacement complete!"
echo "Output written to: $OUTPUT_DIR"
