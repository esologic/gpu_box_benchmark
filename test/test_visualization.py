"""
Test to make sure visualisation happy path works as expected.
"""

import random
import pytest
from typing import Tuple, Optional, List, Iterator, Set
from pathlib import Path
from gpu_box_benchmark.benchmark_types import BenchmarkName, SystemEvaluation
from gpu_box_benchmark import benchmark_types
from gpu_box_benchmark import visualization
from gpu_box_benchmark.visualization import EvaluationsBase, XLabelBase
from test import assets
from _pytest._py.path import LocalPath

_SAMPLE_EVALUATIONS = tuple(
    map(
        benchmark_types.load_system_evaluation_from_disk,
        assets.ALL_SAMPLE_OUTPUTS,
    )
)


def test_create_comparison_visualization(tmpdir: LocalPath) -> None:
    """
    Writes a sample visualization of the test assets to make sure it all works.
    :param tmpdir: Test fixture.
    :return: None
    """

    visualization.create_comparison_visualization(
        evaluations_and_base=EvaluationsBase(
            evaluations=_SAMPLE_EVALUATIONS,
            x_label_base=XLabelBase.gpu,
        ),
        labeled_evaluations=tuple(),
        output_path=Path(tmpdir) / "sample_output.png",
        flatten_versions=True,
        title="A Regular Figure Title!",
        only_render_tests=tuple(),
        n_cols=2,
        watermark=None,
        y_axis_pad_pct=None,
    )


def get_benchmark_iterator(
    num_random_combos: int = 20, seed: int = 42
) -> Iterator[List[BenchmarkName]]:
    """
    Yields every individual benchmark, followed by a reproducible
    sequence of random combinations.
    :param num_random_combos: Number of random combinations.
    :param seed: For `random`.
    :return: Iterator of lists of benchmarks.
    """

    # Initialize the random state for reproducibility
    rng = random.Random(seed)

    # 1. Yield each individual member first
    for bench in BenchmarkName:
        yield [bench]

    # 2. Generate specified number of random combinations
    bench_list = list(BenchmarkName)
    generated: Set[Tuple[BenchmarkName, ...]] = set()

    while len(generated) < num_random_combos:
        # Choose a random length between 2 and the full list length
        r = rng.randint(2, len(bench_list))

        # Pick a random sample and convert to tuple for set storage (hashing)
        combo = tuple(rng.sample(bench_list, r))

        if combo not in generated:
            generated.add(combo)
            yield list(combo)


@pytest.mark.parametrize(
    "evaluations_and_base",
    [
        EvaluationsBase(
            evaluations=_SAMPLE_EVALUATIONS,
            x_label_base=XLabelBase.gpu,
        ),
        EvaluationsBase(
            evaluations=_SAMPLE_EVALUATIONS,
            x_label_base=XLabelBase.cpu,
        ),
        EvaluationsBase(
            evaluations=tuple(),
            x_label_base=XLabelBase.gpu,
        ),
    ],
)
@pytest.mark.parametrize(
    "labeled_evaluations",
    [
        tuple(
            (evaluation, f"label: {index}") for index, evaluation in enumerate(_SAMPLE_EVALUATIONS)
        ),
        tuple(),
    ],
)
@pytest.mark.parametrize(
    "only_render_tests",
    get_benchmark_iterator(num_random_combos=5),
)
@pytest.mark.parametrize(
    "n_cols",
    [
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize(
    "watermark",
    [None, "This is a sample watermark"],
)
@pytest.mark.parametrize(
    "y_axis_pad_pct",
    [
        None,
        0.1,
        0.5,
    ],
)
def test_create_comparison_visualization_exhaustive(  # pylint: disable=too-many-positional-arguments
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    only_render_tests: Tuple[BenchmarkName, ...],
    n_cols: int,
    watermark: Optional[str],
    y_axis_pad_pct: Optional[float],
    tmpdir: LocalPath,
) -> None:
    """
    Parameterizations create an exhaustive set of which options could be passed to visualization
    functions. This will be slow but make sure there are no weird corner case problems.
    :param evaluations_and_base: Passed to viz function.
    :param labeled_evaluations: Passed to viz function.
    :param only_render_tests: Passed to viz function.
    :param n_cols: Passed to viz function.
    :param watermark: Passed to viz function.
    :param y_axis_pad_pct: Passed to viz function.
    :param tmpdir: Test fixture.
    :return: None
    """

    def create_viz() -> Path:
        """
        Wrapper to run the viz creation operation.
        :return: None
        """

        output_path = Path(tmpdir) / "sample_output.png"

        visualization.create_comparison_visualization(
            evaluations_and_base=evaluations_and_base,
            labeled_evaluations=labeled_evaluations,
            output_path=output_path,
            flatten_versions=True,
            title="A Regular Figure Title!",
            only_render_tests=only_render_tests,
            n_cols=n_cols,
            watermark=watermark,
            y_axis_pad_pct=y_axis_pad_pct,
        )

        return output_path

    tests_in_input: Set[BenchmarkName] = visualization.benchmark_names_from_evaluations(
        evaluations=evaluations_and_base.evaluations,
        labeled_evaluations=labeled_evaluations,
    )

    if any(tests_in_input) and any(
        requested_test in tests_in_input for requested_test in only_render_tests
    ):
        assert create_viz().exists()
    else:
        with pytest.raises(ValueError):
            create_viz()
