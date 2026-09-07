import math
from pathlib import Path
from typing import Tuple, Optional, Dict, TypedDict

import pandas as pd

from gpu_box_benchmark.benchmark_types import SystemEvaluation, BenchmarkName, BenchmarkResult, \
    BENCHMARK_TO_PRETTY
from gpu_box_benchmark.visualization import EvaluationsBase, _DFSpecs, \
    _system_evaluations_to_dataframe, render_legend_title, _create_grid, _generate_color_palette, \
    _apply_legend_and_save


class PowerDataFrameSpecs(TypedDict):
    """Structure returned by _system_evaluations_to_power_dataframe"""
    df_throughput: pd.DataFrame
    power_values: dict[str, dict[str, float]]
    specs: dict[str, BenchmarkResult]


def _system_evaluations_to_power_dataframe(
        evaluations_and_base: EvaluationsBase,
        labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
        flatten_versions: bool,
        only_render_tests: Tuple[BenchmarkName, ...],
) -> PowerDataFrameSpecs:
    """
    Extracts Throughput (Y) and Energy (X) data into a aligned dataframe structure.
    """

    # 1. Reuse existing function to get the grid layout (rows/cols) and Y-values (Throughput)
    df_specs: _DFSpecs = _system_evaluations_to_dataframe(
        evaluations_and_base=evaluations_and_base,
        labeled_evaluations=labeled_evaluations,
        flatten_versions=flatten_versions,
        only_render_tests=only_render_tests,
    )

    if df_specs.df.empty:
        raise ValueError("No benchmark data available to plot")

    # 2. Prepare storage for X-values (Energy)
    power_rows: dict[str, dict[str, Optional[float]]] = {}

    # Helper to extract energy from the specific BenchmarkResult object
    def get_energy(benchmark_result: BenchmarkResult) -> Optional[float]:
        gpu_usage = benchmark_result.gpu_power_usage
        critical_key = benchmark_result.critical_gpu_power_usage_key

        if not gpu_usage or not critical_key:
            return None

        key_name = critical_key.value
        # Dynamically access the field on the ReportFileGPUPowerUsage model
        try:
            val = getattr(gpu_usage, key_name)
            return val
        except AttributeError:
            return None

    # Process unlabeled evaluations (reusing your iteration pattern)
    for idx, evaluation in enumerate(evaluations_and_base.evaluations, start=1):
        row_title = render_legend_title(
            index=idx,
            evaluation=evaluation,
            label_base=evaluations_and_base.label_base,
        )
        power_rows[row_title] = {}

        # For every column (benchmark) in our grid...
        for col_name, br_spec in df_specs.specs.items():
            # Find the corresponding BenchmarkResult inside THIS specific evaluation's results list
            matching_result = next((r for r in evaluation.results if r.name == br_spec.name), None)

            if matching_result:
                val = get_energy(matching_result)
                if val is not None:
                    power_rows[row_title][col_name] = val

    # Process labeled evaluations
    for evaluation, row_title in labeled_evaluations:
        power_rows[row_title] = {}
        for col_name, br_spec in df_specs.specs.items():
            matching_result = next((r for r in evaluation.results if r.name == br_spec.name), None)
            if matching_result:
                val = get_energy(matching_result)
                if val is not None:
                    power_rows[row_title][col_name] = val

    return {
        "df_throughput": df_specs.df,
        "power_values": power_rows,
        "specs": df_specs.specs
    }


def create_comparison_visualization_power(
        evaluations_and_base: EvaluationsBase,
        labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
        flatten_versions: bool,
        output_path: Path,
        title: str,
        only_render_tests: Tuple[BenchmarkName, ...],
        n_cols: int,
        watermark: Optional[str],
) -> None:
    """
    Create a comparison visualization with one subplot per benchmark using Scatter plots.

    :param evaluations_and_base: To convert.
    :param labeled_evaluations: To convert.
    :param flatten_versions: Whether to flatten versions into the same plots or keep them separate.
    :param output_path: Where to write the resulting image.
    :param title: Overall figure title.
    :param only_render_tests: Only benchmarks with these names will be included in the output.
    :param n_cols: Number of columns for the subplot grid.
    :param watermark: String written along the right center of the image.
    """

    # 1. Load Data
    data = _system_evaluations_to_power_dataframe(
        evaluations_and_base=evaluations_and_base,
        labeled_evaluations=labeled_evaluations,
        flatten_versions=flatten_versions,
        only_render_tests=only_render_tests,
    )

    df_throughput: pd.DataFrame = data["df_throughput"]
    specs: dict[str, BenchmarkResult] = data["specs"]
    power_values: dict[str, dict[str, float]] = data["power_values"]

    # 2. Setup Grid
    n_plots = len(df_throughput.columns)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = _create_grid(
        n_rows=n_rows,
        n_cols=n_cols,
        sharex=False,
        height_unit=3.0
    )

    # 3. Color Mapping (reusing your existing palette logic)
    system_colors: Dict[str, str] = dict(
        zip(df_throughput.index, _generate_color_palette(len(df_throughput.index)))
    )

    for ax, column in zip(axes, df_throughput.columns):
        spec: BenchmarkResult = specs[column]

        # Extract Y values (Throughput) from existing loader
        series_throughput = df_throughput[column].dropna()

        # Ensure we have a valid dataset to work with
        if series_throughput.empty:
            continue

        # 4. Independent Scaling
        try:
            valid_x_vals = [power_values[sys][column] for sys in series_throughput.index if
                            column in power_values[sys]]
            x_min, x_max = min(valid_x_vals), max(valid_x_vals)
        except KeyError:
            continue

        y_min, y_max = series_throughput.min(), series_throughput.max()

        # Add padding (e.g., 10%) so points don't touch the edges of the plot box
        x_pad = (x_max - x_min) * 0.1 or x_max * 0.1
        y_pad = (y_max - y_min) * 0.1 or y_max * 0.1

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # --- VISUAL FIX: Smaller points (reduced from 200 to 150) ---
        colors = [system_colors[sys] for sys in series_throughput.index]

        ax.scatter(
            valid_x_vals,
            series_throughput.values,
            c=colors,
            edgecolors='k',
            linewidths=0.8,
            zorder=3
        )

        # 5. Styling and Labels
        subtitle = BENCHMARK_TO_PRETTY[BenchmarkName(spec.name)]

        if spec.benchmark_version:
            subtitle = (
                    f"{'↑' if spec.larger_better else '↓'}"
                    + subtitle
                    + ("*" if not spec.multi_gpu_native else "")
            )

        ax.set_title(subtitle, fontsize="medium", pad=10)
        ax.set_ylabel(spec.verbose_unit, fontsize="small")

        # --- VISUAL FIX: Added gridlines to both axes (X and Y) ---
        ax.grid(True, linestyle=":", alpha=0.8)

        ax.set_xlabel("Power Consumption (J)", fontsize="small")

        # 6. Cleanup unused axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    # 7. Save
    _apply_legend_and_save(
        fig=fig,
        label_to_color=system_colors,
        title=title,
        watermark=watermark,
        output_path=output_path,
    )
