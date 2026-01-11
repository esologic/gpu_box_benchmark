#! /usr/bin/env bash

# Create a `venv` virtual environment, activate and install all required packages for development.

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/..

if [[ $(which poetry) ]]; then
  echo "Poetry is installed, continuing."
else
  echo "You need to install Poetry in your system python3.10 See: https://python-poetry.org/docs/#installation"
  read -p "Do you want to install poetry in your system python3.10? (yes/no): " answer
  case $answer in
      [Yy]|[Yy][Ee][Ss])
          curl -sSL https://install.python-poetry.org | python3.10 -
          ;;
      [Nn]|[Nn][Oo])
          echo "Cannot proceed."
          exit
          ;;
      *)
          echo "Invalid input. Please enter 'yes' or 'no'."
          exit
          ;;
  esac
fi

python3.10 -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip

# This doesn't work, need to double check that poetry is in the path!
poetry install --sync --no-ansi
