"""
Core of the comparison between different benchmarks.
"""

from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple, cast

import matplotlib.colors as mcolors
import pandas as pd
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
    SystemEvaluation,
)


class LabelBase(str, Enum):
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


def render_legend_title(index: int, evaluation: SystemEvaluation, label_base: LabelBase) -> str:
    """
    Create a row title for the given evaluation.
    :param index: The index evaluation in the larger batch.
    :param evaluation: To read.
    :param label_base: Selects which field in the evaluation is the basis for the resulting value.
    :return: Depends on `x_label_base`. For gpu: the count of each GPU and type of GPU sorted by
    quantity as a stwing.
    """

    if label_base == LabelBase.gpu:
        return f"#{index}: " + ",".join(
            [
                f"{v}X {k}"
                for k, v in sorted(
                    Counter([gpu.name for gpu in evaluation.gpus]).items(), key=lambda item: item[1]
                )
            ]
        )
    elif label_base == LabelBase.cpu:
        return (
            f"#{index}: " + f"Phys. {evaluation.cpu.physical_cpus}, "
            f"Logi. {evaluation.cpu.logical_cpus} "
            f"- {evaluation.cpu.name}"
        )
    else:
        raise ValueError(f"Unsupported label_base: {label_base}")


def _names_in_evaluation(evaluation: SystemEvaluation) -> Set[BenchmarkName]:
    """
    Helper function to read the names of the tests in the benchmark evaluation.
    :param evaluation: Loaded system evaluation.
    :return: A set of the names of the benchmarks found in the loaded evaluation.
    """

    return set((result.name for result in evaluation.results))


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
    label_base: LabelBase


def benchmark_names_from_evaluations(
    evaluations: Iterable[SystemEvaluation],
    labeled_evaluations: Iterable[Tuple[SystemEvaluation, str]],
) -> Set[BenchmarkName]:
    """
    Library function for reading out the benchmark names from different evaluation sources.
    :param evaluations: Flat list.
    :param labeled_evaluations: Nexted list, the input Tuple is a standard format.
    :return: Set of benchmark names.
    """

    return set().union(
        *[_names_in_evaluation(evaluation=evaluation) for evaluation in evaluations],
        *[
            _names_in_evaluation(evaluation=evaluation)
            for evaluation, _label in labeled_evaluations
        ],
    )


def _create_grid(
    n_rows: int,
    n_cols: int,
    sharex: bool = False,
    height_unit: float = 2.5  # Added optional argument
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Creates a standard sized subplot grid and returns the axes as a flattened list.
    Handles single-subplot cases where plt.subplots doesn't return an array.

    :param n_rows:
    :param n_cols:
    :param sharex:
    :param height_unit: Multiplier for the row height (default 2.5).
    :return:
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

    return fig, axes


def _apply_legend_and_save(
    fig: plt.Figure,
    label_to_color: Dict[str, str],
    title: str,
    watermark: Optional[str],
    output_path: Path,
) -> None:
    """
    Consolidated logic for calculating layout, adding the legend, applying the
    watermark, and saving the figure.
    :param fig: To draw.
    :param label_to_color: Mapping describing how to draw the legend.
    :param title: Super title of the figure.
    :param watermark: Drawn on the right hand side of the output if given.
    :param output_path: Desired output path.
    :return: None
    """

    # Legend Calculation
    legend_handles = [
        plt.Rectangle((0, 0), 0.5, 1, color=color) for color in label_to_color.values()
    ]

    legend_labels: List[str] = list(label_to_color.keys())

    longest_label_length = max((len(label) for label in legend_labels), default=0)
    num_legend_columns = 1 if longest_label_length > 45 else (2 if longest_label_length > 25 else 3)

    resulting_legend = fig.legend(
        legend_handles,
        legend_labels,
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
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


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

    benchmarks_in_evaluations: Set[BenchmarkName] = benchmark_names_from_evaluations(
        evaluations=evaluations_and_base.evaluations,
        labeled_evaluations=labeled_evaluations,
    )

    if any(only_render_tests) and (
        not any((benchmark in benchmarks_in_evaluations for benchmark in only_render_tests))
    ):
        raise ValueError(
            "Benchmarking results did not contain any of the requested tests. "
            f"Requested tests: {only_render_tests} "
            f"Discovered tests: {benchmarks_in_evaluations}"
        )

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

        row_title = render_legend_title(
            index=index,
            evaluation=unlabeled_evaluation,
            label_base=evaluations_and_base.label_base,
        )

        rows[row_title] = column_values_dict(evaluation=unlabeled_evaluation)

    for labeled_evaluation, row_title in labeled_evaluations:
        rows[row_title] = column_values_dict(evaluation=labeled_evaluation)

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Keep specs aligned with dataframe columns
    return _DFSpecs(df=df, specs={col: specs[col] for col in df.columns})


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


def _generate_color_palette(count: int) -> list[str]:
    """
    Deterministic method for getting a color map, a list of colors consumed by matplotlib.
    :param count: How many colors are needed.
    :return: List of colors.
    """

    cmap = plt.get_cmap("Set1")

    # If we need significantly more colors than standard qualitative sets have,
    # switch to a wider palette like 'tab20' or 'Paired'.
    if count > 9:
        cmap = plt.get_cmap("Paired")

    colors = []
    for i in range(count):
        # Map index i (0..N-1) to a value between 0.0 and 1.0
        norm_val = float(i) / max(1, count - 1) if count > 1 else 0.5
        colors.append(mcolors.to_hex(cmap(norm_val)))

    return colors


def create_comparison_visualization(  # pylint: disable=too-many-locals,too-many-positional-arguments
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    flatten_versions: bool,
    output_path: Path,
    title: str,
    only_render_tests: Tuple[BenchmarkName, ...],
    n_cols: int,
    watermark: Optional[str],
    y_axis_pad_pct: Optional[float],
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
    :param y_axis_pad_pct: If given, The y-axis range of each subplot is restricted to this value
    multiplied by the min max values in the y range to clearly show the ranking. If not given,
    auto y-axis settings are used.
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

    # Use helper for figure and grid creation (sharex=False for vertical bar charts)
    fig, axes = _create_grid(n_rows=n_rows, n_cols=n_cols, sharex=False)

    system_colors: Dict[str, str] = dict(
        zip(df_specs.df.index, _generate_color_palette(len(df_specs.df.index)))
    )

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

        ax.set_xticks([])

        subtitle = BENCHMARK_TO_PRETTY[BenchmarkName(spec.name)]
        if spec.benchmark_version:
            subtitle = (
                f"{'↑' if spec.larger_better else '↓'}"
                + subtitle
                + ("*" if not spec.multi_gpu_native else "")
            )

        ax.set_title(subtitle, fontsize="medium")
        ax.set_ylabel(spec.verbose_unit, fontsize="small")

        value_range = series.max() - series.min()

        if y_axis_pad_pct is not None:

            y_axis_numerical_pad_value: float = (
                value_range * y_axis_pad_pct
                if value_range > 0
                else max(abs(series.max()) * y_axis_pad_pct, 1.0)
            )

            ax.set_ylim(
                bottom=series.min() - y_axis_numerical_pad_value,
                top=series.max() + y_axis_numerical_pad_value,
            )

        ax.grid(axis="y", linestyle=":", alpha=0.8)

    # Hide empty plots if we generated more grid slots than we have data
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    # Pass results to the helper for finalizing (legend, watermark, saving)
    _apply_legend_and_save(
        fig=fig,
        label_to_color=system_colors,
        title=title,
        watermark=watermark,
        output_path=output_path,
    )


def create_llm_throughput_visualization(  # pylint: disable=too-many-locals,too-many-nested-blocks
    results: Tuple[LLMThroughputResult, ...],
    plot_model: LLMName,
    output_path: Path,
    title: str,
    watermark: Optional[str],
) -> None:
    """
    Renders the input parsed LLM throughput results as a visualization to the given path.
    :param results: To plot, includes failures.
    :param plot_model: Results may contain more than one model, but the output visualization can
    only contain a single model at a time. Filtering `results` by model occurs within this function.
    :param output_path: Matplotlib visualization is written to this path.
    :param title: Added to the top of the figure.
    :param watermark: Written to the right of the figure.
    :return: None
    """

    model_results: Tuple[LLMThroughputResult, ...] = tuple(
        result for result in results if result.model == plot_model
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

    fig, axes = _create_grid(n_rows=len(context_depths), n_cols=len(quants), sharex=True)

    unique_labels = {result.label for result in model_results}

    label_to_color: Dict[str, str] = dict(
        zip(unique_labels, _generate_color_palette(len(unique_labels)))
    )

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
            for label_index, label in enumerate(reversed(label_to_color.keys())):

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
                    color=label_to_color[label],
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
        label_to_color=label_to_color,
        title=title,
        watermark=watermark,
        output_path=output_path,
    )
