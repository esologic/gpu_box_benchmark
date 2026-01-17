#!/usr/bin/env bash

# Run isort in check mode to ensure that there are not outstanding changes that need to be made.
# Will exit non-zero if there are errors or incorrectly formatted python imports.

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..
source ./.venv/bin/activate

export PYTHONPATH=".:./test${PYTHONPATH+:}${PYTHONPATH:-}"

# If this is failing, replace the line with `isort -c -vb` to see verbose output
isort ./gpu_box_benchmark bench_cli.py --check --diff
