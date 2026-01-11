#! /usr/bin/env bash

# Run all code checks to make sure code meets formatting standards.
# Will exit non-zero if there are errors or improperly formatted code.

set -euo pipefail
FAILED=""

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Running Pylint"
$DIR/run_pylint.sh || FAILED="pylint $FAILED"

echo "Running Isort"
$DIR/run_isort.sh || FAILED="isort $FAILED"

echo "Running Black"
$DIR/run_black.sh || FAILED="black $FAILED"

echo "Running Mypy"
$DIR/run_mypy.sh || FAILED="mypy $FAILED"

if [ -n "$FAILED" ]; then
  echo "FAILED: $FAILED"
  exit 1
else
  echo "All Succeeded"
fi
