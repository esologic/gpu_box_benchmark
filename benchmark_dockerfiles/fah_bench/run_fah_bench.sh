#!/bin/bash

# Set defaults if env vars are not set
FAHBENCH_PRECISION="${FAHBENCH_PRECISION:-single}"

exec FAHBench-cmd \
    --platform OpenCL \
    --precision "$FAHBENCH_PRECISION" \
    --run-length 60
