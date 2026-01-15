#!/bin/bash
set -euo pipefail

: "${RUN_ENV:?RUN_ENV is required}"

# Parse shell-quoted argv (trusted input only)
eval "ARGS=($RUN_ENV)"

exec ./benchmark-launcher-cli benchmark "${ARGS[@]}"
