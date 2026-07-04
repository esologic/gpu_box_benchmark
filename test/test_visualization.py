"""
Test to make sure visualisation happy path works as expected.
"""

from pathlib import Path

from gpu_box_benchmark import benchmark_types
from gpu_box_benchmark import visualization
from gpu_box_benchmark.visualization import EvaluationsBase, XLabelBase
from test.assets import ALL_SAMPLE_OUTPUTS
from _pytest._py.path import LocalPath


def test_create_comparison_visualization(tmpdir: LocalPath) -> None:
    """
    Writes a sample visualization of the test assets to make sure it all works.
    :param tmpdir: Test fixture.
    :return: None
    """

    visualization.create_comparison_visualization(
        evaluations_and_base=EvaluationsBase(
            evaluations=tuple(
                map(
                    benchmark_types.load_system_evaluation_from_disk,
                    ALL_SAMPLE_OUTPUTS,
                )
            ),
            x_label_base=XLabelBase.gpu,
        ),
        labeled_evaluations=tuple(),
        output_path=Path(tmpdir) / "sample_output.png",
        flatten_versions=True,
        title="yes",
        only_render_tests=tuple(),
        n_cols=2,
        watermark=None,
        y_axis_pad_pct=None,
    )
