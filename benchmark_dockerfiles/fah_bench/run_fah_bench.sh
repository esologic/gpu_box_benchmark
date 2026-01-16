#!/bin/bash

# Set defaults if env vars are not set
FAHBENCH_PRECISION="${FAHBENCH_PRECISION:-single}"
FAHBENCH_RUN_LENGTH="${FAHBENCH_RUN_LENGTH:-60}"  # in seconds

exec FAHBench-cmd \
    --platform OpenCL \
    --precision "$FAHBENCH_PRECISION" \
    --run-length "$FAHBENCH_RUN_LENGTH"
