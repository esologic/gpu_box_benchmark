"""
Core of the comparison between different benchmarks.
"""

import math
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd

from gpu_box_benchmark.benchmark_jobs import BENCHMARK_TO_PRETTY, BenchmarkName
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult, SystemEvaluation


def _benchmark_column_name(result: BenchmarkResult) -> str:
    """
    Formats the `BenchmarkResult.name` column name form.
    :param result: To read.
    :return: Column name as a string.
    """
    return f"{result.name}@{result.benchmark_version}"


def _extract_critical_value(result: BenchmarkResult) -> float | None:
    """
    Extract the benchmark's headline numerical value.
    :param result: To read.
    :return: The critical value.
    """
    key = result.critical_result_key.value
    return cast(float | None, getattr(result.numerical_results, key))


def _create_title(index: int, evaluation: SystemEvaluation) -> str:
    """
    Create a row title for the given evaluation.
    :param index: The index evaluation in the larger batch.
    :param evaluation: To read.
    :return: The count of each GPU and type of GPU sorted by quantity as a stwing.
    """

    return f"#{index}: " + ",".join(
        [
            f"{v}X {k}"
            for k, v in sorted(
                Counter([gpu.name for gpu in evaluation.gpus]).items(), key=lambda item: item[1]
            )
        ]
    )


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


def _system_evaluations_to_dataframe(evaluations: Tuple[SystemEvaluation, ...]) -> _DFSpecs:
    """
    Convert a list of SystemEvaluation objects into a wide dataframe.

    :param evaluations: To convert.
    :return:
        - DataFrame: index = system title, columns = benchmarks
        - Dict mapping column name -> representative BenchmarkResult (spec)
    """
    rows: dict[str, dict[str, float]] = {}
    specs: dict[str, BenchmarkResult] = {}

    for index, evaluation in enumerate(evaluations, start=1):
        system_row: dict[str, float] = {}

        for result in evaluation.results:
            column = _benchmark_column_name(result)

            # Record spec once (first occurrence wins)
            specs.setdefault(column, result)

            # We could parameterize this later.
            value = _extract_critical_value(result)

            # Allow missing / unsupported results to stay NaN
            if value is not None:
                system_row[column] = value

        rows[_create_title(index=index, evaluation=evaluation)] = system_row

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Stable, deterministic ordering
    df = df.sort_index(axis=0).sort_index(axis=1)

    # Keep specs aligned with dataframe columns
    return _DFSpecs(df=df, specs={col: specs[col] for col in df.columns})


def create_comparison_visualization(  # pylint: disable=too-many-locals
    evaluations: Tuple[SystemEvaluation, ...],
    output_path: Path,
    title: str,
    n_cols: int = 4,
    max_bar_label_length: int = 15,
) -> None:
    """
    Create a comparison visualization with one subplot per benchmark.

    :param evaluations: System evaluations to compare.
    :param output_path: Where to write the resulting image.
    :param n_cols: Number of columns to show.
    :param max_bar_label_length: For the x-axis bars, this is the max number of characters to show
    before truncation.
    :param title: Overall figure title.
    """

    df_specs = _system_evaluations_to_dataframe(evaluations)

    if df_specs.df.empty:
        raise ValueError("No benchmark data available to plot")

    n_plots = len(df_specs.df.columns)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(12 * n_cols / 3, 5 * n_rows),
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

        if series.min() >= 0:
            ax.set_ylim(bottom=0)

        ax.grid(axis="y", linestyle=":", alpha=0.8)

    for ax, column in zip(axes[:n_plots], df_specs.df.columns):
        series = df_specs.df[column].dropna()
        labels = list(series.index)

        if ax is not axes[n_plots - 1]:
            labels = [_truncate_label(lbl, max_bar_label_length) for lbl in labels]

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")

    fig.suptitle(title, fontsize="large")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
