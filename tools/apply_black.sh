#!/usr/bin/env bash

# Run Black - change code to make sure that python code is properly formatted.

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..
source ./.venv/bin/activate

black ./bench_cli.py .
