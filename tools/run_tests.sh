#!/usr/bin/env bash

# Run pytest unit tests in the ./gpu_box_benchmark:./test directory.
# Will exit non-zero if any unit test fail unexpectedly.
#
# Pass in other pytest args, such as for very verbose test output:
#   export PYTEST_EXTRA_ARGS="-vv"
#
# The optional first argument is the pytest marker expression.
# The default is 'all' tests. Example:
#   ./tools/run_tests.sh "not integration"
#
# Any additional arguments are passed directly to pytest:
#   ./tools/run_tests.sh all test/test_run_benchmarks.py -s

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${DIR}/.."
source ./.venv/bin/activate

export PYTHONPATH=".:./test${PYTHONPATH+:}${PYTHONPATH:-}"

TEST_MARKED="${1:-all}"
if [[ -n "${1:-}" ]]; then
    shift
fi

PYTEST_ARGS=()

if [[ -n "${PYTEST_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    PYTEST_ARGS+=(${PYTEST_EXTRA_ARGS})
fi

echo "run tests: ${TEST_MARKED}; args: ${PYTEST_ARGS[*]:-} $*"

pytest -m "${TEST_MARKED/#all/}" "${PYTEST_ARGS[@]}" "$@"