#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0,1}
TRAIN_DATA=${TRAIN_DATA:-data/train.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PYTHON_BIN" -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --gpu_ids "$GPU_IDS" --mixed_precision bf16 \
  -m train_multi_llm.train \
  --train-data "$TRAIN_DATA" --multi-model-path "$OUTPUT_ROOT/multi/final" \
  --output-dir "$OUTPUT_ROOT/multi_llm" "$@"
