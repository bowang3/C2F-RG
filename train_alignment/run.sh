#!/usr/bin/env bash
set -euo pipefail

# Run from github/ with paths supplied by environment variables.
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-0,1}
MODEL_ROOT=${MODEL_ROOT:-models}
TRAIN_DATA=${TRAIN_DATA:-data/alignment.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-output}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

common=(--train-data "$TRAIN_DATA" --clinfusion-path "$MODEL_ROOT/ClinFusion-8B"
  --base-model-path "$MODEL_ROOT/Qwen3-VL-8B-Instruct"
  --dinov2-path "$MODEL_ROOT/dinov2-large"
  --convnext-path "$MODEL_ROOT/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup")
launch=(--multi_gpu --num_processes 2 --gpu_ids "$GPU_IDS" --mixed_precision bf16)

"$PYTHON_BIN" -m accelerate.commands.launch "${launch[@]}" -m train_alignment.train \
  "${common[@]}" --stage a1 --epochs 1 --learning-rate 1e-5 \
  --output-dir "$OUTPUT_ROOT/alignment_a1" "$@"

"$PYTHON_BIN" -m accelerate.commands.launch "${launch[@]}" -m train_alignment.train \
  "${common[@]}" --stage a2 --epochs 3 --learning-rate 5e-6 --pe3d-learning-rate 1e-6 \
  --a1-checkpoint "$OUTPUT_ROOT/alignment_a1/final" \
  --output-dir "$OUTPUT_ROOT/alignment_a2" "$@"
