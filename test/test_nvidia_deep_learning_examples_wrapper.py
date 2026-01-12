"""
Test of container running and output parsing.
"""

from pathlib import Path
from test.assets import RESNET50_OUTPUT_PATH

import pytest

from gpu_box_benchmark import nvidia_deep_learning_examples_wrapper
from gpu_box_benchmark.nvidia_deep_learning_examples_wrapper import ReportFileNumerical


@pytest.mark.parametrize(
    "report_file,expected_result",
    [
        (
            RESNET50_OUTPUT_PATH,
            ReportFileNumerical(
                sample_count=31.0,
                mean=27.360552390625422,
                std=0.4914936826285199,
                result_min=24.71908042452736,
                percentile_25=27.43164908499008,
                percentile_50=27.44542666231421,
                percentile_75=27.471883145928565,
                result_max=27.51118586993748,
            ),
        ),
    ],
)
def test_parse_report_file(report_file: Path, expected_result: ReportFileNumerical) -> None:
    """
    Test to make sure we can parse some known output in a report file.
    :param report_file: Path to test asset.
    :param expected_result: Expected result, a computation on the parsed data.
    :return: None
    """

    assert nvidia_deep_learning_examples_wrapper.parse_report_file(report_file) == expected_result
