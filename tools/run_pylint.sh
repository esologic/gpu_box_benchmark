#! /usr/bin/env bash

# Run pylint - make sure python code is correctly styled, is able to check for things that aren't
# looked for by black.
# Will exit non-zero if there are errors or incorrectly formatted python code.

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..
source ./.venv/bin/activate

export PYTHONPATH=".:./test${PYTHONPATH+:}${PYTHONPATH:-}"

pylint main.py gpu_box_benchmark test
