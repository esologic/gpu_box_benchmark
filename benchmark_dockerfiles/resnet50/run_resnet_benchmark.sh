#!/bin/bash
set -euo pipefail

# Environment variables
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"
MODE_TRAINING="${MODE_TRAINING:-1}"  # 1 = training, 0 = inference

# Set mode-specific flag
if [ "$MODE_TRAINING" = "1" ]; then
    MODE_FLAG="--training-only"

else
    MODE_FLAG="--evaluate"
fi

# Run torchrun
torchrun \
  --nproc_per_node="$NUM_GPUS" \
  main.py \
    --arch resnet50 \
    --no-checkpoints \
    --run-epochs 1 \
    --prof 300 \
    --workers 5 \
    --data-backend synthetic \
    "$MODE_FLAG" \
    -b "$BATCH_SIZE" \
    --raport-file /results/result.txt \
    /workspace/dataset
