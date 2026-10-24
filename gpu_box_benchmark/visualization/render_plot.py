"""
Core of the comparison between different benchmarks.
"""

import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt

from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench import (
    llama_bench_dockerfile_wrapper,
)
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench.llama_bench_types import (
    LLMName,
    LLMThroughputResult,
)
from gpu_box_benchmark.benchmark_types import (
    BENCHMARK_TO_PRETTY,
    BenchmarkName,
    BenchmarkResult,
    LineMode,
    SystemEvaluation,
)
from gpu_box_benchmark.visualization.load_for_visualization import (
    DFsSpecs,
    EvaluationsBase,
    LoadedLLMResults,
    system_evaluations_to_dataframe,
)

_DEFAULT_DPI = 300


def _create_grid(
    n_rows: int,
    n_cols: int,
    dpi: int,
    sharex: bool = False,
    height_unit: float = 2.5,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Creates a standard sized subplot grid and returns the axes as a flattened list.
    Handles single-subplot cases where `plt.subplots` doesn't return an array.

    :param n_rows: Rows of subplots.
    :param n_cols: Columns of subplots.
    :param dpi: Sets the figure's internal DPI. This ensures: 1. Vector backends (SVG) use
    a high-res viewbox by default2. Raster calculations (in _apply_legend_and_save) match the
    export density.
    :param sharex: Passed to call.
    :param height_unit: Multiplier for the row height (default 2.5).
    :return: Figure and axes list.
    """

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(3 * n_cols, height_unit * n_rows),
        sharex=sharex,
    )

    if isinstance(axes, plt.Axes):
        # Single subplot case returns an Axes object directly
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_dpi(dpi)

    return fig, axes


def _patch_svg_pixel_size(path: Path, width_px: int, height_px: int) -> None:
    """
    Rewrite the root <svg> width/height; the viewBox keeps content scaling correct.
    :param path: File to modify.
    :param width_px: New width.
    :param height_px: New height.
    :return: None
    """

    text = path.read_text(encoding="utf-8")
    text = re.sub(r'(<svg\b[^>]*?\swidth=")[^"]*(")', rf"\g<1>{width_px}px\g<2>", text, count=1)
    text = re.sub(r'(<svg\b[^>]*?\sheight=")[^"]*(")', rf"\g<1>{height_px}px\g<2>", text, count=1)

    path.write_text(text, encoding="utf-8")


def _apply_legend_and_save(  # pylint: disable=too-many-positional-arguments,too-many-locals
    fig: plt.Figure,
    ordered_labels: List[str],
    label_to_color: Dict[str, str],
    title: str,
    watermark: Optional[str],
    legend_columns: Optional[int],
    output_path: Path,
    dpi: int,
) -> None:
    """
    Consolidated logic for calculating layout, adding the legend, applying the
    watermark, and saving the figure.
    :param fig: To draw.
    :param label_to_color: Mapping describing how to draw the legend.
    :param title: Super title of the figure.
    :param watermark: Drawn on the right hand side of the output if given.
    :param legend_columns: Manual override for the number of columns in the output.
    :param output_path: Desired output path.
    :return: None
    """

    # Legend Calculation
    legend_handles = [
        plt.Rectangle((0, 0), 0.5, 1, color=label_to_color[color]) for color in ordered_labels
    ]

    longest_label_length = max((len(label) for label in ordered_labels), default=0)
    num_legend_columns = (
        (1 if longest_label_length > 45 else (2 if longest_label_length > 25 else 3))
        if legend_columns is None
        else legend_columns
    )

    resulting_legend = fig.legend(
        legend_handles,
        ordered_labels,
        loc="outside lower center",
        ncol=num_legend_columns,
        frameon=True,
        facecolor="#f5f5f5",
        edgecolor="lightgrey",
        fontsize="small",
    )

    # Layout Calculation (Positioning tight_layout based on legend height)
    legend_bbox = resulting_legend.get_window_extent(
        fig.canvas.get_renderer()  # type: ignore[attr-defined]
    )

    legend_bbox_fig = legend_bbox.transformed(fig.transFigure.inverted())
    tight_layout_bottom = max(legend_bbox_fig.y1, 0)

    if watermark:
        fig.text(
            0.995,  # Standardized x-position (pinned to right edge with slight padding)
            0.5,
            watermark,
            fontsize="small",
            color="gray",
            alpha=0.6,
            ha="right",
            va="center",
            rotation="vertical",
        )

    # Apply Suptitle and Layout
    fig.suptitle(title, fontsize="large")

    # Using 0.96 on the right to prevent layout overlap issues with watermark/text
    fig.tight_layout(pad=1.5, w_pad=2, rect=(0, tight_layout_bottom, 0.96, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)

    if output_path.suffix.lower() == ".svg":
        w_in, h_in = fig.get_size_inches()
        _patch_svg_pixel_size(output_path, round(w_in * dpi), round(h_in * dpi))

    plt.close(fig)


def _draw_bar(
    x_axis_max: float,
    ax: plt.Axes,
    label_index: int,
    color: str,
    result: Optional[LLMThroughputResult],
) -> None:
    """
    Helper to draw the specific visualization state.
    :param x_axis_max: Used to locate the text on the bar.
    :param ax: Axis to draw on.
    :param label_index: Index to draw the bar to within the subplot.
    :param color: Bar color.
    :param result: To plot.
    :return: None
    """

    if result is not None:
        if result.tokens_per_second is not None:
            # Successful Execution
            ax.barh(label_index, result.tokens_per_second, color=color)
            ax.text(
                result.tokens_per_second + x_axis_max * 0.01,
                label_index,
                f"{result.tokens_per_second:.1f}",
                ha="left",
                va="center",
                fontsize="x-small",
            )
        else:
            # Failed Execution (Marked by white bar + hatching)
            ax.barh(label_index, x_axis_max, color="white", edgecolor=color, hatch="///")
            ax.text(
                x_axis_max / 2,
                label_index,
                "FAILED",
                ha="center",
                va="center",
                fontsize="small",
                fontweight="bold",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.2"},
            )
    else:
        # No Data (Marked by white bar + hatching)
        ax.barh(label_index, x_axis_max, color="white", edgecolor=color, hatch="///")
        ax.text(
            x_axis_max / 2,
            label_index,
            "NO DATA",
            ha="center",
            va="center",
            fontsize="small",
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.2"},
        )


def _format_subplot_title(spec: BenchmarkResult, max_width: int = 35) -> str:
    """
    Constructs the formatted string for the subplot subtitle with automatic wrapping.

    :param spec: Benchmark result to create the title for.
    :param max_width: Maximum number of characters per line (default 25).
    :return: Title string, potentially containing newlines.
    """
    base = BENCHMARK_TO_PRETTY[BenchmarkName(spec.name)]

    if spec.benchmark_version:
        direction = "↑" if spec.larger_better else "↓"
        modifier = "*" if not spec.multi_gpu_native else ""
        base = f"{direction}{base}{modifier}"

    base = "\n".join(textwrap.wrap(base, width=max_width))

    return base


def _get_padded_bounds(
    data_min: float, data_max: float, pad_pct: Optional[float]
) -> tuple[float, float]:
    """
    Calculates plot limits with consistent padding relative to the data's scale.

    Uses the maximum magnitude of the values as the base for padding. This ensures
    that a 10% padding always represents roughly the same visual amount
    across different benchmarks (e.g. Tokens/sec vs FPS), regardless of the
    absolute range or minimum values.

    :param data_min: Min value in the data being plotted.
    :param data_max: Max value in the data being plotted.
    :param pad_pct: Pad percent. Used to add some space in resulting plots. If `None`, returns
    exact data bounds to allow auto-scaling.
    :return: Bounds.
    """

    # When padding is not specified, return exact bounds so matplotlib handles scaling
    if pad_pct is None:
        return data_min, data_max

    pad_amount = (
        max(abs(data_max), abs(data_min)) if (data_max != 0 or data_min != 0) else 1.0
    ) * pad_pct

    return data_min - pad_amount, data_max + pad_amount


def _find_result(
    model_results: Tuple[LLMThroughputResult, ...],
    label: str,
    context_depth: int,
    quantization: str,
) -> Optional[LLMThroughputResult]:
    """
    Helper to find the relevant result for a specific cell.
    :param model_results: Search space.
    :param label: Needs to match output.
    :param context_depth: Needs to match output.
    :param quantization: Needs to match output.
    :return: Result if found, None if otherwise.
    """

    return next(
        (
            r
            for r in model_results
            if r.label == label
            and r.context_depth == context_depth
            and r.quantization == quantization
        ),
        None,
    )


def create_system_comparison_visualization_results(  # pylint: disable=too-many-locals, too-many-positional-arguments, too-many-arguments
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    flatten_versions: bool,
    output_path: Path,
    title: str,
    only_render_tests: Tuple[BenchmarkName, ...],
    n_cols: int,
    watermark: Optional[str],
    legend_columns: Optional[int],
    y_axis_pad_pct: Optional[float] = None,
    dpi: int = _DEFAULT_DPI,
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
    to show before truncation.
    :param watermark: String written along the right center of the image.
    :param legend_columns: Override for the number of columns to include in the legend.
    :param y_axis_pad_pct: If given, The y-axis range of each subplot is restricted to this value
    multiplied by the min max values in the y range to clearly show the ranking. If not given,
    auto y-axis settings are used.
    :param dpi: Dots per inch of output image.
    :return: None
    """

    dfs_specs: DFsSpecs = system_evaluations_to_dataframe(
        evaluations_and_base=evaluations_and_base,
        labeled_evaluations=labeled_evaluations,
        flatten_versions=flatten_versions,
        only_render_tests=only_render_tests,
    )

    if dfs_specs.df_results.empty:
        raise ValueError("No benchmark data available to plot")

    n_plots = len(dfs_specs.df_results.columns)
    n_rows = math.ceil(n_plots / n_cols)

    # Use helper for figure and grid creation (sharex=False for vertical bar charts)
    fig, axes = _create_grid(n_rows=n_rows, n_cols=n_cols, sharex=False, dpi=dpi)

    for ax, column in zip(axes, dfs_specs.df_results.columns):
        series = dfs_specs.df_results[column].dropna()
        spec = dfs_specs.specs[column]

        x_ticks = range(len(series))
        ax.bar(
            x_ticks,
            series.values,
            color=[dfs_specs.color_mapping[system] for system in series.index],
            width=1,
        )

        ax.set_xticks([])

        # Use shared subtitle helper
        ax.set_title(_format_subplot_title(spec), fontsize="medium")
        ax.set_ylabel(spec.verbose_unit, fontsize="small")

        lower, upper = _get_padded_bounds(
            data_min=series.min(), data_max=series.max(), pad_pct=y_axis_pad_pct
        )
        ax.set_ylim(bottom=lower, top=upper)

        ax.grid(axis="y", linestyle=":", alpha=0.8)

    # Hide empty plots if we generated more grid slots than we have data
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    # Pass results to the helper for finalizing (legend, watermark, saving)
    _apply_legend_and_save(
        fig=fig,
        ordered_labels=dfs_specs.label_order,
        label_to_color=dfs_specs.color_mapping,
        title=title,
        watermark=watermark,
        legend_columns=legend_columns,
        output_path=output_path,
        dpi=dpi,
    )


def create_system_comparison_visualization_results_power(  # pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
    line_mode: LineMode,
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    flatten_versions: bool,
    output_path: Path,
    title: str,
    only_render_tests: Tuple[BenchmarkName, ...],
    n_cols: int,
    watermark: Optional[str],
    legend_columns: Optional[int],
    y_axis_pad_pct: float = 0.1,
    dpi: int = _DEFAULT_DPI,
) -> None:
    """
    Creates a scatter plot per input evalution. X-axis is power consumption, Y-axis is the units
    for the actual test.

    :param line_mode: Controls whether and how points within each group are connected:
        LineMode.no_lines does not draw connecting lines.
        LineMode.input_order connects points in their original group position order.
        LineMode.x_value sorts points within each group by X value before connecting them.
    :param evaluations_and_base: To convert.
    :param labeled_evaluations: To convert.
    :param flatten_versions: Whether to flatten versions into the same plots or keep them separate.
    :param output_path: Where to write the resulting image.
    :param title: Overall figure title.
    :param only_render_tests: Only benchmarks with these names will be included in the output
    visualization.
    :param n_cols: Subplot columns
    :param watermark: String written along the right center of the image.
    :param legend_columns: Override for the number of columns to include in the legend.
    :param y_axis_pad_pct: If given, The y-axis range of each subplot is restricted to this value
    multiplied by the min max values in the y range to clearly show the ranking. If not given,
    auto y-axis settings are used.
    :param dpi: Dots per inch of the output image.
    :return: None
    """

    dfs_specs: DFsSpecs = system_evaluations_to_dataframe(
        evaluations_and_base=evaluations_and_base,
        labeled_evaluations=labeled_evaluations,
        flatten_versions=flatten_versions,
        only_render_tests=only_render_tests,
    )

    n_plots = len(dfs_specs.df_results.columns)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = _create_grid(n_rows=n_rows, n_cols=n_cols, sharex=False, height_unit=3, dpi=dpi)

    dataframe_indices = list(dfs_specs.df_results.index)
    grouped_evaluations = evaluations_and_base.grouped_evaluations

    # Only build the group mapping when grouped inputs were actually provided.
    # Regular --input evaluations have no corresponding group.
    index_to_group: Dict[str, str] = {
        dataframe_index: group
        for dataframe_index, (_, group) in zip(
            dataframe_indices,
            grouped_evaluations,
        )
    }

    for ax, column in zip(axes, dfs_specs.df_results.columns):
        spec: BenchmarkResult = dfs_specs.specs[column]
        series_results = dfs_specs.df_results[column].dropna()

        if series_results.empty:
            continue

        # Filter for valid power data to ensure X/Y alignment
        valid_mask = dfs_specs.df_power.loc[series_results.index, column].notna()
        valid_sys = series_results.index[valid_mask]

        # Convert power from Joules to Kilojoules for accurate scaling
        valid_x_vals = [v / 1000 for v in dfs_specs.df_power.loc[valid_sys, column]]

        if not valid_x_vals:
            continue

        # Synchronize Y-values and colors with X-filtering to prevent shape mismatches
        valid_series = series_results[valid_mask]
        colors = [dfs_specs.color_mapping[sys] for sys in valid_sys]

        x_min, x_max = min(valid_x_vals), max(valid_x_vals)
        y_min, y_max = valid_series.min(), valid_series.max()

        # Apply padding using shared helper
        x_lower, x_upper = _get_padded_bounds(
            x_min,
            x_max,
            pad_pct=0.1,
        )
        y_lower, y_upper = _get_padded_bounds(
            y_min,
            y_max,
            pad_pct=y_axis_pad_pct,
        )

        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_lower, y_upper)

        ax.scatter(
            valid_x_vals,
            valid_series.values,
            c=colors,
            edgecolors="k",
            linewidths=0.8,
            zorder=3,
            alpha=0.8,
        )

        if grouped_evaluations and line_mode != LineMode.no_lines:

            # Build the points for each group. The final integer preserves the
            # original position so we can optionally use group order instead
            # of X order.
            grouped_points: Dict[str, List[Tuple[float, float, str, int]]] = {}

            for position, (sys, x_val, y_val) in enumerate(
                zip(
                    valid_sys,
                    valid_x_vals,
                    valid_series.values,
                )
            ):
                group = index_to_group[sys]
                color = dfs_specs.color_mapping[sys]

                grouped_points.setdefault(group, []).append(
                    (
                        x_val,
                        y_val,
                        color,
                        position,
                    )
                )

            for group_points in grouped_points.values():
                if len(group_points) < 2:
                    continue

                if line_mode == LineMode.x_value:
                    group_points.sort(key=lambda point: point[0])
                else:
                    group_points.sort(key=lambda point: point[3])

                group_x = [point[0] for point in group_points]
                group_y = [point[1] for point in group_points]
                group_color = group_points[0][2]

                ax.plot(
                    group_x,
                    group_y,
                    linestyle="--",
                    linewidth=1.2,
                    color=group_color,
                    alpha=0.7,
                    zorder=2,
                )

        # Use shared subtitle helper
        ax.set_title(
            _format_subplot_title(spec),
            fontsize="medium",
            pad=10,
        )
        ax.set_ylabel(spec.verbose_unit, fontsize="small")
        ax.set_xlabel("Power Consumption (kJ)", fontsize="small")

        ax.grid(True, linestyle=":", alpha=0.8)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    _apply_legend_and_save(
        fig=fig,
        ordered_labels=dfs_specs.label_order,
        label_to_color=dfs_specs.color_mapping,
        title=title,
        watermark=watermark,
        legend_columns=legend_columns,
        output_path=output_path,
        dpi=dpi,
    )


def create_llm_throughput_visualization(  # pylint: disable=too-many-locals,too-many-nested-blocks,too-many-positional-arguments
    results: LoadedLLMResults,
    plot_model: LLMName,
    output_path: Path,
    title: str,
    watermark: Optional[str],
    legend_columns: Optional[int],
    dpi: int = _DEFAULT_DPI,
) -> None:
    """
    Renders the input parsed LLM throughput results as a visualization to the given path.
    :param results: To plot, includes failures.
    :param plot_model: Results may contain more than one model, but the output visualization can
    only contain a single model at a time. Filtering `results` by model occurs within this function.
    :param output_path: Matplotlib visualization is written to this path.
    :param title: Added to the top of the figure.
    :param watermark: Written to the right of the figure.
    :param legend_columns: Override for the number of columns to include in the legend.
    :param dpi: Sets the dots per inch of the output image.
    :return: None
    """

    model_results: Tuple[LLMThroughputResult, ...] = tuple(
        result for result in results.results if result.model == plot_model
    )

    if not model_results:
        raise ValueError(f"No benchmark data available for {plot_model.value}")

    # Use SUPPORTED_QUANTS to ensure canonical ordering (e.g., q4 before q8)
    quants = tuple(
        q
        for q in llama_bench_dockerfile_wrapper.SUPPORTED_QUANTS
        if any(r.quantization == q for r in model_results)
    )

    # Use SUPPORTED_CONTEXTS values (integers) to ensure canonical ordering
    context_depths = tuple(
        v
        for v in llama_bench_dockerfile_wrapper.SUPPORTED_CONTEXTS.values()
        if any(r.context_depth == v for r in model_results)
    )

    fig, axes = _create_grid(n_rows=len(context_depths), n_cols=len(quants), sharex=True, dpi=dpi)

    subplot_title_text_size = "medium"

    # Safe scaling logic (ignores None/failed results)
    max_tokens_per_second: float = max(
        (
            result.tokens_per_second
            for result in model_results
            if result.tokens_per_second is not None
        ),
        default=0.0,
    )

    x_axis_max = max(1.0, max_tokens_per_second * 1.15)

    # Main Grid Loop
    for context_index, context_depth in enumerate(context_depths):
        for quantization_index, quantization in enumerate(quants):

            ax = axes[context_index * len(quants) + quantization_index]
            ax.set_xlim(0, x_axis_max)

            # Iterate through systems and delegate drawing to the closure

            # Reversed here so the top to bottom reading matches the left to right reading of the
            # legend.
            for label_index, label in enumerate(reversed(results.label_order)):

                result: Optional[LLMThroughputResult] = _find_result(
                    model_results=model_results,
                    label=label,
                    context_depth=context_depth,
                    quantization=quantization,
                )

                _draw_bar(
                    x_axis_max=x_axis_max,
                    ax=ax,
                    label_index=label_index,
                    color=results.color_mapping[label],
                    result=result,
                )

            # Axis Styling & Labels
            ax.set_yticks([])
            ax.set_title(quantization, fontsize=subplot_title_text_size)
            ax.grid(axis="x", linestyle=":", alpha=0.8)

            if quantization_index == 0:
                ax.text(
                    -0.08,
                    0.5,
                    f"{context_depth // 1024}k",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=subplot_title_text_size,
                )

            if context_index == len(context_depths) - 1:
                ax.set_xlabel("Tokens / Second", fontsize="small")

    fig.supylabel("Context Depth")

    _apply_legend_and_save(
        fig=fig,
        ordered_labels=results.label_order,
        label_to_color=results.color_mapping,
        title=title,
        watermark=watermark,
        legend_columns=legend_columns,
        output_path=output_path,
        dpi=dpi,
    )
