"""Uses pathlib to make referencing test assets by path easier."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

ASSETS_DIRECTORY_PATH = str(_ASSETS_DIRECTORY)
