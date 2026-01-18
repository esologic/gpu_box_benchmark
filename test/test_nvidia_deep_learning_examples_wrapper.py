"""
Test of container running and output parsing.
"""

from pathlib import Path
from test.assets import RESNET50_OUTPUT_PATH

import pytest

from gpu_box_benchmark.benchmark_dockerfile_wrappers import nvidia_deep_learning_examples
from gpu_box_benchmark.docker_wrapper import ContainerOutputs


@pytest.mark.parametrize(
    "report_file,expected_result",
    [
        (RESNET50_OUTPUT_PATH, 27.360552390625422),
    ],
)
def test__parse_report_file(report_file: Path, expected_result: float) -> None:
    """
    Test to make sure we can parse some known output in a report file.
    :param report_file: Path to test asset.
    :param expected_result: Expected result, a computation on the parsed data.
    :return: None
    """

    assert (
        nvidia_deep_learning_examples._parse_report_file(  # pylint: disable=protected-access
            container_output=ContainerOutputs(logs="", file=report_file), mode_training=True
        )
        == expected_result
    )
