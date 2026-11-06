#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0,1}
MODEL_ROOT=${MODEL_ROOT:-models}
TRAIN_DATA=${TRAIN_DATA:-data/train.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$PYTHON_BIN" -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --gpu_ids "$GPU_IDS" --mixed_precision bf16 \
  -m train_multi.train \
  --train-data "$TRAIN_DATA" --aligned-model-path "$OUTPUT_ROOT/alignment_a2/final" \
  --base-model-path "$MODEL_ROOT/Qwen3-VL-8B-Instruct" \
  --dinov2-path "$MODEL_ROOT/dinov2-large" \
  --convnext-path "$MODEL_ROOT/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup" \
  --output-dir "$OUTPUT_ROOT/multi" "$@"
