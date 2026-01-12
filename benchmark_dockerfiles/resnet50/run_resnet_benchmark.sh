#!/bin/bash
set -euo pipefail

# ---------------------------
# Environment variables
# ---------------------------
BATCH_SIZE="${BATCH_SIZE:-32}"
NGPUS="${NGPUS:-1}"
AMP="${AMP:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MODE_TRAINING="${MODE_TRAINING:-1}"  # 1 = training, 0 = inference

# ---------------------------
# Set mode-specific flag
# ---------------------------
if [ "$MODE_TRAINING" = "1" ]; then
    MODE_FLAG="--training-only"
else
    MODE_FLAG="--evaluate"
fi

# ---------------------------
# Run torchrun
# ---------------------------
torchrun \
  --nproc_per_node="$NGPUS" \
  main.py \
    --arch resnet50 \
    --no-checkpoints \
    --run-epochs 1 \
    --prof 300 \
    --data-backend synthetic \
    $MODE_FLAG \
    --workers "$NUM_WORKERS" \
    -b "$BATCH_SIZE" \
    $( [ "$AMP" = "1" ] && echo "--amp" ) \
    --raport-file /results/output.txt \
    /workspace/dataset
