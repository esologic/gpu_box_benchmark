#!/usr/bin/env bash

# Apply isort - change code to make sure that python imports are properly sorted

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..
source ./.venv/bin/activate

export PYTHONPATH=".:./test${PYTHONPATH+:}${PYTHONPATH:-}"

isort .
