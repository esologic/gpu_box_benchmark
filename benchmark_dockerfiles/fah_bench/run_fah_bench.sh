#!/bin/bash
# -----------------------------------------------------------------------------
# run_fah_bench.sh
# Wrapper to run FAHBench with options controlled entirely via environment variables.
# -----------------------------------------------------------------------------

# Set defaults if env vars are not set
FAHBENCH_PRECISION="${FAHBENCH_PRECISION:-single}"
FAHBENCH_PLATFORM="${FAHBENCH_PLATFORM:-CUDA}"
FAHBENCH_DEVICE_ID="${FAHBENCH_DEVICE_ID:-0}"
FAHBENCH_RUN_LENGTH="${FAHBENCH_RUN_LENGTH:-60}"  # in seconds

# Optional: extra arguments can still be passed to the script
EXTRA_ARGS="$@"

echo "Running FAHBench with the following configuration:"
echo "  Platform   : $FAHBENCH_PLATFORM"
echo "  Device ID  : $FAHBENCH_DEVICE_ID"
echo "  Precision  : $FAHBENCH_PRECISION"
echo "  Run length : $FAHBENCH_RUN_LENGTH s"
echo "  Extra args : $EXTRA_ARGS"

exec FAHBench-cmd \
    --platform "$FAHBENCH_PLATFORM" \
    --precision "$FAHBENCH_PRECISION" \
    --run-length "$FAHBENCH_RUN_LENGTH" \
    $EXTRA_ARGS
