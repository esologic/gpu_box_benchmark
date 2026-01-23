"""Uses pathlib to make referencing test assets by path easier."""

from pathlib import Path

_ASSETS_DIRECTORY = Path(__file__).parent.resolve()

RESNET50_OUTPUT_PATH = _ASSETS_DIRECTORY / "resnet50_output.txt"

SAMPLE_OUTPUT_1_PATH = _ASSETS_DIRECTORY / "sample_output_1.json"
SAMPLE_OUTPUT_2_PATH = _ASSETS_DIRECTORY / "sample_output_2.json"
SAMPLE_OUTPUT_3_PATH = _ASSETS_DIRECTORY / "sample_output_3.json"


ALL_SAMPLE_OUTPUTS = (SAMPLE_OUTPUT_1_PATH, SAMPLE_OUTPUT_2_PATH, SAMPLE_OUTPUT_3_PATH)
