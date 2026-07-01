"""
Core of the comparison between different benchmarks.
"""

import math
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd

from gpu_box_benchmark.benchmark_types import (
    BENCHMARK_TO_PRETTY,
    BenchmarkName,
    BenchmarkResult,
    SystemEvaluation,
)


class XLabelBase(str, Enum):
    """
    Options for the basis of the labels on the x axis of the graphs in the resulting visualizations.
    """

    gpu = "gpu"
    cpu = "cpu"


def _benchmark_column_name(result: BenchmarkResult, flatten_version: bool) -> str:
    """
    Formats the `BenchmarkResult.name` column name form.
    :param result: To read.
    :param flatten_version: Whether to include the version number.
    :return: Column name as a string.
    """
    return result.name if flatten_version else f"{result.name}@{result.benchmark_version}"


def _extract_critical_value(result: BenchmarkResult) -> float | None:
    """
    Extract the benchmark's headline numerical value.
    :param result: To read.
    :return: The critical value.
    """
    key = result.critical_result_key.value
    return cast(float | None, getattr(result.numerical_results, key))


def _row_title_from_base(index: int, evaluation: SystemEvaluation, x_label_base: XLabelBase) -> str:
    """
    Create a row title for the given evaluation.
    :param index: The index evaluation in the larger batch.
    :param evaluation: To read.
    :param x_label_base: Selects which field in the evaluation is the basis for the resulting value.
    :return: Depends on `x_label_base`. For gpu: the count of each GPU and type of GPU sorted by
    quantity as a stwing.
    """

    if x_label_base == XLabelBase.gpu:
        return f"#{index}: " + ",".join(
            [
                f"{v}X {k}"
                for k, v in sorted(
                    Counter([gpu.name for gpu in evaluation.gpus]).items(), key=lambda item: item[1]
                )
            ]
        )
    elif x_label_base == XLabelBase.cpu:
        return (
            f"#{index}: " + f"Phys. {evaluation.cpu.physical_cpus}, "
            f"Logi. {evaluation.cpu.logical_cpus} "
            f"- {evaluation.cpu.name}"
        )
    else:
        raise ValueError(f"Unsupported x_label_base: {x_label_base}")


def _truncate_label(text: str, max_len: int) -> str:
    """
    Truncates the given text to the length, adds an ellipsis.
    :param text: To modify.
    :param max_len: Max length inclusive.
    :return: Truncated label.
    """

    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class _DFSpecs(NamedTuple):
    """
    Links the dataframe with a sample dict mapping column name to one of the benchmarks to be able
    to pull out units, versions etc.
    """

    df: pd.DataFrame
    specs: dict[str, BenchmarkResult]


class EvaluationsBase(NamedTuple):
    """
    Intermediate type. Links a set of evaluations with the distinguishing element of the run that
    should be used as the basis for the resulting x-axis label.
    """

    evaluations: Tuple[SystemEvaluation, ...]
    x_label_base: XLabelBase


def _system_evaluations_to_dataframe(
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    flatten_versions: bool,
    only_render_tests: Tuple[BenchmarkName, ...],
) -> _DFSpecs:
    """
    Convert a list of SystemEvaluation objects into a wide dataframe.

    :param evaluations_and_base: NT containing a list of Evaluations and an enum that describes
    what the x-axis label should be.
    :param labeled_evaluations: Iterator of the loaded evaluations and their specific x-axis label.
    :param flatten_versions: Whether to flatten versions into the same plots or keep them separate.
    :param only_render_tests: Only benchmarks with these names will be included in the ouput
    visualization.
    :return: A tuple containing
        - DataFrame: index = system title, columns = benchmarks
        - Dict mapping column name -> representative BenchmarkResult (spec)
    """
    rows: dict[str, dict[str, float]] = {}
    specs: dict[str, BenchmarkResult] = {}

    def column_values_dict(evaluation: SystemEvaluation) -> dict[str, float]:
        """
        Helper function to create the dictionary that will become the values for the column values.
        :param evaluation: To turn into a row.
        :return: Row for the given evaluation.
        """

        output: dict[str, float] = {}

        benchmark_results: Iterable[BenchmarkResult] = evaluation.results

        if any(only_render_tests):
            benchmark_results = filter(lambda b_r: b_r.name in only_render_tests, benchmark_results)

        for benchmark_result in benchmark_results:

            column = _benchmark_column_name(benchmark_result, flatten_version=flatten_versions)

            # Record spec once (first occurrence wins)
            specs.setdefault(column, benchmark_result)

            # We could parameterize this later.
            value = _extract_critical_value(benchmark_result)

            # Allow missing / unsupported results to stay NaN
            if value is not None:
                output[column] = value

        return output

    for index, unlabeled_evaluation in enumerate(evaluations_and_base.evaluations, start=1):
        row_title = _row_title_from_base(
            index=index,
            evaluation=unlabeled_evaluation,
            x_label_base=evaluations_and_base.x_label_base,
        )

        rows[row_title] = column_values_dict(evaluation=unlabeled_evaluation)

    for labeled_evaluation, row_title in labeled_evaluations:
        rows[row_title] = column_values_dict(evaluation=labeled_evaluation)

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Keep specs aligned with dataframe columns
    return _DFSpecs(df=df, specs={col: specs[col] for col in df.columns})


def create_comparison_visualization(  # pylint: disable=too-many-locals,too-many-positional-arguments
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    flatten_versions: bool,
    output_path: Path,
    title: str,
    only_render_tests: Tuple[BenchmarkName, ...],
    n_cols: int = 4,
    max_bar_label_length_chars: Optional[int] = 15,
) -> None:
    """
    Create a comparison visualization with one subplot per benchmark.

    :param evaluations_and_base: To convert.
    :param labeled_evaluations: To convert.
    :param flatten_versions: Whether to flatten versions into the same plots or keep them separate.
    :param output_path: Where to write the resulting image.
    :param title: Overall figure title.
    :param only_render_tests: Only benchmarks with these names will be included in the output
    visualization.
    :param n_cols: Number of columns to show.
    :param max_bar_label_length_chars: For the x-axis bars, this is the max number of characters
    to show before truncation.
    :return: None
    """

    df_specs: _DFSpecs = _system_evaluations_to_dataframe(
        evaluations_and_base=evaluations_and_base,
        labeled_evaluations=labeled_evaluations,
        flatten_versions=flatten_versions,
        only_render_tests=only_render_tests,
    )

    if df_specs.df.empty:
        raise ValueError("No benchmark data available to plot")

    n_plots = len(df_specs.df.columns)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8 * n_cols, 5 * n_rows),
        sharex=False,
    )

    # Flatten axes for easy iteration
    axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    system_colors = {
        system: color_cycle[i % len(color_cycle)] for i, system in enumerate(df_specs.df.index)
    }

    for ax, column in zip(axes, df_specs.df.columns):
        series = df_specs.df[column].dropna()
        spec = df_specs.specs[column]

        x_ticks = range(len(series))
        ax.bar(
            x_ticks,
            series.values,
            color=[system_colors[system] for system in series.index],
            width=1,
        )
        ax.set_xticks(x_ticks)

        subtitle = BENCHMARK_TO_PRETTY[BenchmarkName(spec.name)]
        if spec.benchmark_version:
            subtitle = (
                f"{'↑' if spec.larger_better else '↓'}"
                + subtitle
                + ("*" if not spec.multi_gpu_native else "")
            )

        ax.set_title(subtitle, fontsize="medium")
        ax.set_ylabel(spec.verbose_unit)

        value_range = series.max() - series.min()

        padding: float = value_range * 0.2 if value_range > 0 else max(abs(series.max()) * 0.2, 1.0)

        ax.set_ylim(
            bottom=max(0.0, series.min() - padding),
            top=series.max() + padding,
        )

        ax.grid(axis="y", linestyle=":", alpha=0.8)

    for ax, column in zip(axes[:n_plots], df_specs.df.columns):
        series = df_specs.df[column].dropna()
        labels = list(series.index)

        if ax is not axes[n_plots - 1]:
            labels = [
                (
                    _truncate_label(lbl, max_bar_label_length_chars)
                    if max_bar_label_length_chars
                    else lbl
                )
                for lbl in labels
            ]

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize="large")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
