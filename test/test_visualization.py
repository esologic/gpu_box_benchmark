"""
Test to make sure visualisation happy path works as expected.
"""

from pathlib import Path
from gpu_box_benchmark import numeric_benchmark_result
from gpu_box_benchmark import visualization
from test.assets import ALL_SAMPLE_OUTPUTS
from _pytest._py.path import LocalPath


def test_create_comparison_visualization(tmpdir: LocalPath) -> None:
    """
    Writes a sample visualization of the test assets to make sure it all works.
    :param tmpdir: Test fixture.
    :return: None
    """

    visualization.create_comparison_visualization(
        evaluations=tuple(
            map(numeric_benchmark_result.load_system_evaluation_from_disk, ALL_SAMPLE_OUTPUTS)
        ),
        output_path=Path(tmpdir) / "sample_output.png",
        title="yes",
    )
