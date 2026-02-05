#!/bin/bash
set -e

# Configuration
FILE_SIZE="5G"
TEST_FILE="/app/model_benchmark.bin"
WORKERS=$(nproc)

# Default to Read (0) if not provided. Write is 1.
IO_TYPE=${IO_TYPE:-0}

gdsio -f $TEST_FILE -d 0 -s $FILE_SIZE -i 1M -x 2 -I $IO_TYPE -w $WORKERS