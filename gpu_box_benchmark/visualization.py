"""
Core of the comparison between different benchmarks.
"""

from pathlib import Path
from typing import NamedTuple, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd

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

    for system in evaluations:
        system_row: dict[str, float] = {}

        for result in system.results:
            column = _benchmark_column_name(result)

            # Record spec once (first occurrence wins)
            specs.setdefault(column, result)

            # We could parameterize this later.
            value = _extract_critical_value(result)

            # Allow missing / unsupported results to stay NaN
            if value is not None:
                system_row[column] = value

        rows[system.title] = system_row

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Stable, deterministic ordering
    df = df.sort_index(axis=0).sort_index(axis=1)

    # Keep specs aligned with dataframe columns
    return _DFSpecs(df=df, specs={col: specs[col] for col in df.columns})


def create_comparison_visualization(
    evaluations: Tuple[SystemEvaluation, ...],
    output_path: Path,
    title: str,
) -> None:
    """
    Create a comparison visualization with one subplot per benchmark.

    :param evaluations: System evaluations to compare.
    :param output_path: Where to write the resulting image.
    :param title: Overall figure title.
    """

    df_specs = _system_evaluations_to_dataframe(evaluations)

    if df_specs.df.empty:
        raise ValueError("No benchmark data available to plot")

    n_plots = len(df_specs.df.columns)

    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        sharex=True,
        figsize=(12, 5 * n_plots),
    )

    if n_plots == 1:
        axes = [axes]

    # --- NEW: stable color per system ---
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    system_colors = {
        system: color_cycle[i % len(color_cycle)] for i, system in enumerate(df_specs.df.index)
    }

    for ax, column in zip(axes, df_specs.df.columns):
        series = df_specs.df[column].dropna()
        spec = df_specs.specs[column]

        colors = [system_colors[system] for system in series.index]

        ax.bar(series.index, series.values, color=colors, width=1)

        subtitle = spec.name
        if spec.benchmark_version:
            subtitle += f" @ {spec.benchmark_version} {'↑' if spec.larger_better else '↓'}"
        ax.set_title(subtitle, loc="left", fontsize="medium")

        ax.set_ylabel(spec.unit or "score")

        if series.min() >= 0:
            ax.set_ylim(bottom=0)

        ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.xticks(rotation=45, ha="right")

    fig.suptitle(title, fontsize="large")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
