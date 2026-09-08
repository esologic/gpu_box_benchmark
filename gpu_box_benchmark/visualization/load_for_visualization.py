"""
Code related to loading the data to be visualized.
"""

from collections import Counter
from enum import Enum
from typing import Iterable, NamedTuple, Optional, Set, Tuple, cast

import pandas as pd

from gpu_box_benchmark.benchmark_types import BenchmarkName, BenchmarkResult, SystemEvaluation


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

    evaluations: Tuple[SystemEvaluation, ...]
    label_base: LabelBase


class DFsSpecs(NamedTuple):
    """
    Holds the aligned dataframes for results and power consumption,
    along with a mapping of column names to their representative BenchmarkResult specs.
    """

    df_results: pd.DataFrame
    df_power: pd.DataFrame
    specs: dict[str, BenchmarkResult]


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

    results_rows: dict[str, dict[str, float]] = {}
    power_rows: dict[str, dict[str, Optional[float]]] = {}
    specs: dict[str, BenchmarkResult] = {}

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
            column: str = _benchmark_column_name(benchmark_result, flatten_version=flatten_versions)

            # Record spec once (first occurrence wins)
            specs.setdefault(column, benchmark_result)

            # Use the new extraction function to get both values at once
            data = _extract_benchmark_data(benchmark_result)

            if data.result_value is not None:
                output_row_results[column] = data.result_value

            if data.power_value is not None:
                output_row_power[column] = data.power_value

        return output_row_results, output_row_power

    # Process unlabeled evaluations
    for idx, evaluation in enumerate(evaluations_and_base.evaluations, start=1):
        row_title = render_legend_title(
            index=idx,
            evaluation=evaluation,
            label_base=evaluations_and_base.label_base,
        )
        results_row, power_row = process_evaluation_results(
            benchmark_results_iter=evaluation.results
        )
        results_rows[row_title] = results_row
        power_rows[row_title] = power_row

    # Process labeled evaluations
    for evaluation, row_title in labeled_evaluations:
        results_row, power_row = process_evaluation_results(
            benchmark_results_iter=evaluation.results
        )
        results_rows[row_title] = results_row
        power_rows[row_title] = power_row

    # Construct DataFrames
    df_results = pd.DataFrame.from_dict(results_rows, orient="index")

    if not power_rows:
        # If no power data was collected, create an empty dataframe aligned to the result rows
        df_power = pd.DataFrame(index=df_results.index)
    else:
        # Ensure power dataframe columns align perfectly with results dataframe
        df_power = pd.DataFrame.from_dict(power_rows, orient="index").reindex(
            columns=df_results.columns
        )

    return DFsSpecs(df_results=df_results, df_power=df_power, specs=specs)
