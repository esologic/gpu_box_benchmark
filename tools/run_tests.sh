#!/usr/bin/env bash

# Run all pytest unit tests in the ./gpu_box_benchmark:./test directory.
# Will exit non-zero if any unit test fail unexpectedly.
#
# Pass in other pytest args, such as for very verbose test output:
#   export PYTEST_EXTRA_ARGS=-vv
#
# On the command line, the optional first argument is the expression to select marked tests.
# The default is 'all' tests, an example expression is "not integration".
# Use an empty string or the special value 'all' for all tests.
# Any additional arguments, directories and filenames are passed to pytest.

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..
source ./.venv/bin/activate

export PYTHONPATH=".:./test${PYTHONPATH+:}${PYTHONPATH:-}"

TEST_MARKED=${1:-all}
if [ "${1:-}" ]; then shift 1; fi

echo "run tests: ${TEST_MARKED}; args: ${PYTEST_EXTRA_ARGS:-} $@"

pytest -m "${TEST_MARKED/#all/}" "${PYTEST_EXTRA_ARGS:-}" "$@"
