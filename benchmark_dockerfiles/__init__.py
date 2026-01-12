"""Allows us to easily locate assets within this directory from python code."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

RESNET50_DOCKERFILE = _ASSETS_DIRECTORY / "resnet50" / "Dockerfile"
