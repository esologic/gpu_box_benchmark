"""
Code related to loading the data to be visualized.
"""

import colorsys
import itertools
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench import (
    llama_bench_dockerfile_wrapper,
)
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench.llama_bench_types import (
    LLMThroughputResult,
)
from gpu_box_benchmark.benchmark_types import (
    BenchmarkName,
    BenchmarkResult,
    SystemEvaluation,
    load_system_evaluation_from_disk,
)


class SystemComparisonPlotMode(str, Enum):
    """
    Different supported system comparison plot modes
    """

    results = "results"
    results_power = "results_power"


class LabelBase(str, Enum):
    """
    Options for the basis of the labels on the x axis of the graphs in the resulting visualizations.
    """

    gpu = "gpu"
    cpu = "cpu"


class EvaluationsBase(NamedTuple):
    """
    Intermediate type. Links a set of evaluations with the distinguishing element of the run that
    should be used as the basis for the resulting x-axis label.
    """

    ungrouped_evaluations: Tuple[SystemEvaluation, ...]
    grouped_evaluations: Tuple[Tuple[SystemEvaluation, str], ...]
    label_base: LabelBase


class DFsSpecs(NamedTuple):
    """
    Holds the aligned dataframes for results and power consumption,
    along with a mapping of column names to their representative BenchmarkResult specs.
    """

    df_results: pd.DataFrame
    df_power: pd.DataFrame
    specs: Dict[str, BenchmarkResult]
    label_order: List[str]

    color_mapping: Dict[str, str]
    """
    Maps label to it's color in the resulting viz. Used to establish groups. 
    """


class BenchmarkData(NamedTuple):
    """
    Stores both the headline numerical value and the corresponding power consumption.
    """

    result_value: Optional[float]
    power_value: Optional[float]


def _benchmark_column_name(result: BenchmarkResult, flatten_version: bool) -> str:
    """
    Formats the `BenchmarkResult.name` column name form.
    :param result: To read.
    :param flatten_version: Whether to include the version number.
    :return: Column name as a string.
    """
    return result.name if flatten_version else f"{result.name}@{result.benchmark_version}"


def _extract_benchmark_data(result: BenchmarkResult) -> BenchmarkData:
    """
    Pull out the critical results from a given benchmark result.
    :param result: To read.
    :return: NT containing the possible critical results.
    """

    # Extract the headline numerical value (Y-axis)
    key = result.critical_result_key.value
    value = cast(float | None, getattr(result.numerical_results, key))

    # Extract power/energy data (X-axis)
    gpu_usage = result.gpu_power_usage
    critical_key = result.critical_gpu_power_usage_key
    power_value: float | None = None

    if gpu_usage and critical_key:
        try:
            power_value = getattr(gpu_usage, critical_key.value)
        except AttributeError:
            pass

    return BenchmarkData(result_value=value, power_value=power_value)


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


def _get_group_shades(hue: float, saturation: float, count: int) -> List[str]:
    """
    Generate 'count' distinct hex colors based on a base hue and saturation.

    We vary the lightness around a safe center (0.65) to ensure all bars
    are visible and clearly distinguishable within their group.
    :param hue: The hue component (0.0 - 1.0).
    :param saturation: The saturation component (0.0 - 1.0).
    :param count: Number of distinct shades needed for this group.
    :return: List of hex color strings.
    """
    shades = []
    target_l = 0.65

    if count == 1:
        # Single item gets the full vibrant color at the center lightness
        l_var = max(0.25, min(0.85, target_l))
        shades.append(mcolors.to_hex(colorsys.hls_to_rgb(hue, l_var, saturation)))
    else:
        # Spread lightness evenly around the center to prevent "muddy" or too-dark colors
        step = 0.4 / (count - 1)  # Span of roughly 0.4 in lightness space
        center_idx = (count - 1) / 2

        for i in range(count):
            l_var = target_l + (i - center_idx) * step

            # Clamp to prevent washout (near white) or near-black bars
            l_var = max(0.25, min(0.85, l_var))

            shades.append(mcolors.to_hex(colorsys.hls_to_rgb(hue, l_var, saturation)))

    return shades


def generate_color_mapping(
    all_labels: List[str], group_to_labels: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Create a dictionary mapping labels to their color in the plot.
    Ensures colors are visually distinct and groups share similar hues
    but different lightnesses.

    :param all_labels: Contains labels in/out of groups.
    :param group_to_labels: Labels within the given group should have similar colors.
    :return: Dictionary mapping each label to a hex color string.
    """

    # Output to populate
    result_colors: Dict[str, str] = {}

    # Assign distinct base hues to each group.
    num_groups = len(group_to_labels)

    if num_groups > 0:
        for index, labels in enumerate(group_to_labels.values()):

            # Evenly distribute groups around the color wheel.
            hue = float(index) / float(num_groups)

            # Use a consistent, reasonably strong saturation.
            # HLS lightness is controlled by _get_group_shades.
            saturation = 0.65

            # Delegate the shading logic to the helper function.
            shades = _get_group_shades(
                hue=hue,
                saturation=saturation,
                count=len(labels),
            )

            for label, color in zip(labels, shades):
                result_colors[label] = color

    # Handle ungrouped labels using a qualitative map.
    ungrouped_labels = [label for label in all_labels if label not in result_colors]

    ungrouped_colormap = plt.get_cmap("Set1")

    for index, label in enumerate(ungrouped_labels):
        normalization_constant = (
            float(index) / max(1, len(ungrouped_labels) - 1) if len(ungrouped_labels) > 1 else 0.5
        )

        result_colors[label] = mcolors.to_hex(
            cast(
                Tuple[float, float, float],
                tuple(ungrouped_colormap(normalization_constant)[0:3]),
            )
        )

    return result_colors


def system_evaluations_to_dataframe(  # pylint: disable=too-many-locals
    evaluations_and_base: EvaluationsBase,
    labeled_evaluations: Tuple[Tuple[SystemEvaluation, str], ...],
    flatten_versions: bool,
    only_render_tests: Tuple[BenchmarkName, ...],
) -> DFsSpecs:
    """
    Convert a list of SystemEvaluation objects into aligned wide dataframes for results and power.

    :param evaluations_and_base: NT containing a list of Evaluations and an enum that describes
    what the x-axis label should be.
    :param labeled_evaluations: Iterator of the loaded evaluations and their specific x-axis label.
    :param flatten_versions: Whether to flatten versions into the same plots or keep them separate.
    :param only_render_tests: Only benchmarks with these names will be included in the output
    visualization.
    :return: A tuple containing (df_results, df_power, specs).
    """

    benchmarks_in_evaluations: set[BenchmarkName] = benchmark_names_from_evaluations(
        evaluations=(
            evaluations_and_base.ungrouped_evaluations
            + tuple(evaluation for evaluation, label in evaluations_and_base.grouped_evaluations)
        ),
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

    results_rows: dict[str, dict[str, float]] = {}
    power_rows: dict[str, dict[str, Optional[float]]] = {}
    specs: dict[str, BenchmarkResult] = {}
    label_order: List[str] = []

    def process_evaluation_results(
        benchmark_results_iter: Iterable[BenchmarkResult],
    ) -> tuple[dict[str, float], dict[str, Optional[float]]]:
        """
        Converts the results to rows.
        :param benchmark_results_iter: Data source.
        :return: Tuple of rows.
        """

        output_row_results: dict[str, float] = {}
        output_row_power: dict[str, Optional[float]] = {}

        if any(only_render_tests):
            benchmark_results_iter = filter(
                lambda b_r: b_r.name in only_render_tests, benchmark_results_iter
            )

        for benchmark_result in benchmark_results_iter:
            column: str = _benchmark_column_name(
                benchmark_result,
                flatten_version=flatten_versions,
            )

            # Record spec once (first occurrence wins)
            specs.setdefault(column, benchmark_result)

            # Use the new extraction function to get both values at once
            data = _extract_benchmark_data(benchmark_result)

            if data.result_value is not None:
                output_row_results[column] = data.result_value

            if data.power_value is not None:
                output_row_power[column] = data.power_value

        return output_row_results, output_row_power

    def fill_evaluation_results(evaluation: SystemEvaluation, title: str) -> str:
        """
        Populate the lists for processing.
        :param evaluation: To read.
        :param title: Label.
        :return: Input label forwarded.
        """

        results_row, power_row = process_evaluation_results(
            benchmark_results_iter=evaluation.results
        )

        results_rows[title] = results_row
        power_rows[title] = power_row

        if title not in label_order:
            label_order.append(title)

        return title

    # Process unlabeled, ungrouped evaluations
    for idx, evaluation in enumerate(evaluations_and_base.ungrouped_evaluations, start=1):
        fill_evaluation_results(
            evaluation=evaluation,
            title=render_legend_title(
                index=idx,
                evaluation=evaluation,
                label_base=evaluations_and_base.label_base,
            ),
        )

    group_to_labels: Dict[str, List[str]] = defaultdict(list)

    # Process unlabeled, grouped evaluations
    for idx, (evaluation, group) in enumerate(evaluations_and_base.grouped_evaluations, start=1):
        group_to_labels[group].append(
            fill_evaluation_results(
                evaluation=evaluation,
                title=render_legend_title(
                    index=idx,
                    evaluation=evaluation,
                    label_base=evaluations_and_base.label_base,
                ),
            )
        )

    # Process labeled evaluations
    for evaluation, row_title in labeled_evaluations:

        row_results, row_power = process_evaluation_results(
            benchmark_results_iter=evaluation.results
        )

        results_rows[row_title] = row_results
        power_rows[row_title] = row_power

        if row_title not in label_order:
            label_order.append(row_title)

    # Construct DataFrames
    df_results = pd.DataFrame.from_dict(results_rows, orient="index").reindex(index=label_order)

    if only_render_tests:
        valid_columns = [col for col in only_render_tests if col in df_results.columns]
        if valid_columns:
            df_results = df_results[valid_columns]

    if not power_rows:
        # If no power data was collected, create an empty dataframe aligned to the result rows
        df_power = pd.DataFrame(index=label_order)
    else:
        # Ensure power dataframe rows and columns align perfectly with results dataframe
        # (Because we reordered df_results above, df_power will pick up the correct order here)
        df_power = pd.DataFrame.from_dict(power_rows, orient="index").reindex(
            index=label_order,
            columns=df_results.columns,
        )

    return DFsSpecs(
        df_results=df_results,
        df_power=df_power,
        specs=specs,
        label_order=label_order,
        color_mapping=generate_color_mapping(
            all_labels=label_order, group_to_labels=group_to_labels
        ),
    )


class LoadedLLMResults(NamedTuple):
    """
    Output of the load and EXPLODE of the `LLMThroughputResult` data. Need to preserve the
    original label order from the input.
    """

    label_order: List[str]
    results: Tuple[LLMThroughputResult, ...]
    color_mapping: Dict[str, str]


def cli_inputs_to_llm_results(
    input_paths: Tuple[Path, ...],
    grouped_inputs: Tuple[Tuple[Path, str], ...],
    legend_label: LabelBase,
    labeled_input: Tuple[Tuple[Path, str], ...],
) -> LoadedLLMResults:
    """
    Helper function to take user input and generate a set of NTs for plotting.
    :param input_paths: List of input evaluations on disk.
    :param grouped_inputs: Inputs with their corresponding labels from the CLI.
    :param legend_label: Decides how plots are labeled in the legend.
    :param labeled_input: Pre-labeled inputs from the CLI, user wants a specific label for the
    given input evaluation.
    :return: NTs for plotting.
    """

    results: Tuple[LLMThroughputResult, ...] = tuple(
        itertools.chain.from_iterable(
            [
                llama_bench_dockerfile_wrapper.read_throughput_results(
                    complete_results=evaluation.results,
                    failed_tests=evaluation.failed_tests,
                    label=render_legend_title(
                        index=label_index, evaluation=evaluation, label_base=legend_label
                    ),
                )
                for label_index, evaluation in enumerate(
                    map(load_system_evaluation_from_disk, input_paths), start=1
                )
            ]
        )
    )

    group_to_labels: Dict[str, List[str]] = defaultdict(list)

    for label_index, (evaluation_path, group) in enumerate(grouped_inputs, start=1):

        evaluation = load_system_evaluation_from_disk(path=evaluation_path)

        label = render_legend_title(
            index=label_index, evaluation=evaluation, label_base=legend_label
        )

        results += tuple(
            llama_bench_dockerfile_wrapper.read_throughput_results(
                complete_results=evaluation.results,
                failed_tests=evaluation.failed_tests,
                label=label,
            )
        )

        group_to_labels[group].append(label)

    results += tuple(
        itertools.chain.from_iterable(
            (
                llama_bench_dockerfile_wrapper.read_throughput_results(
                    complete_results=labeled_evaluation.results,
                    failed_tests=labeled_evaluation.failed_tests,
                    label=label,
                )
                for labeled_evaluation, label in [
                    (load_system_evaluation_from_disk(path=path), label)
                    for path, label in labeled_input
                ]
            )
        )
    )

    label_order = []

    for result in results:
        if result.label not in label_order:
            label_order.append(result.label)

    return LoadedLLMResults(
        label_order=label_order,
        results=results,
        color_mapping=generate_color_mapping(
            all_labels=label_order, group_to_labels=group_to_labels
        ),
    )
