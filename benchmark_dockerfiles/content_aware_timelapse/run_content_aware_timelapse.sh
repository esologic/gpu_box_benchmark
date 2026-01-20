#!/bin/bash
set -euo pipefail

# Parse shell-quoted argv (trusted input only)
eval "ARGS=($RUN_ENV)"

/app/content_aware_timelapse/.venv/bin/python catcli.py benchmark "${ARGS[@]}"