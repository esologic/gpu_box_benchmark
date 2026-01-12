"""Uses pathlib to make referencing test assets by path easier."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

RESNET50_OUTPUT_PATH = _ASSETS_DIRECTORY / "resnet50_output.txt"
