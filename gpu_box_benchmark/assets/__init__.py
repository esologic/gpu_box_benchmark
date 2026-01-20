"""Uses pathlib to make referencing assets by path easier."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

SYSTEMD_TEMPLATE_PATH = _ASSETS_DIRECTORY / "gpu_box_benchmark.service.jinja2"
