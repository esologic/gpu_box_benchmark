"""Main module."""

import itertools
import logging
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import psutil
from bonus_click import options
from click.decorators import FC

from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench import (
    llama_bench_dockerfile_wrapper,
)
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench.llama_bench_types import (
    LLMName,
    LLMThroughputResult,
)
from gpu_box_benchmark.benchmark_types import (
    BENCHMARK_TO_PRETTY,
    EXTENDED_BENCHMARK_DOCUMENTS,
    BenchmarkFamily,
    BenchmarkName,
    BenchmarkResult,
    SystemEvaluation,
    load_system_evaluation_from_disk,
)
from gpu_box_benchmark.cli_common import (
    BENCHMARK_COMMAND_NAME,
    ENV_VAR_MAPPING,
    create_default_output_name,
)
from gpu_box_benchmark.executor_lookup import render_named_executors
from gpu_box_benchmark.locate_describe_hardware import (
    GPU_CLICK_OPTION,
    GPUIdentity,
    discover_cpu,
    discover_gpus,
)
from gpu_box_benchmark.render_systemd import render_systemd_file
from gpu_box_benchmark.visualization import render_plot
from gpu_box_benchmark.visualization.load_for_visualization import (
    EvaluationsBase,
    LabelBase,
    SystemComparisonPlotMode,
    render_legend_title,
)

LOGGER_FORMAT = "[%(asctime)s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)

_GPU_BOX_BENCHMARK_VERSION = "0.3.0"
"""
See top level `CHANGELOG.md` for version history of this format.
"""


def run_options() -> Callable[[FC], FC]:
    """
    Creates the group of click options that define a run.
    :return: Wrapped command.
    """

    def output(command: FC) -> FC:
        """
        Wrap the input command.
        :param command: To wrap.
        :return: Wrapped input.
        """

        decorators = [
            click.option(
                "--output-parent",
                type=click.Path(
                    exists=True,
                    file_okay=False,
                    dir_okay=True,
                    writable=True,
                    readable=True,
                    path_type=Path,
                ),
                default=Path("./").resolve(),
                show_default=True,
                help="The resulting system evaluation is written into this directory",
                envvar=ENV_VAR_MAPPING["output_parent"],
                show_envvar=True,
            ),
            click.option(
                "--title",
                type=click.STRING,
                help=(
                    "A short few words to describe the run."
                    " Will be a plot title in resulting visualizations."
                ),
                default="GPU Box Test",
                show_default=True,
                envvar=ENV_VAR_MAPPING["title"],
                show_envvar=True,
            ),
            click.option(
                "--description",
                type=click.STRING,
                help="Longer text field to qualitatively describe the run in a more verbose way.",
                default="Sample Description. This run was completed on a computer made of corn!",
                show_default=True,
                envvar=ENV_VAR_MAPPING["description"],
                show_envvar=True,
            ),
        ]

        for dec in reversed(decorators):
            dec(command)

        return command

    return output


@click.group()
def cli() -> None:
    """
    Uses Docker to run GPU related benchmarks with the goal of comparing system hardware
    architectures.

    \f

    :return: None
    """


@cli.command(
    name=BENCHMARK_COMMAND_NAME, short_help="Run one or more benchmarks and records the results."
)
@options.create_enum_option(
    "--test",
    help_message="Decides which benchmark to run.",
    input_enum=BenchmarkName,
    multiple=True,
)
@GPU_CLICK_OPTION
@click.option(
    "--output-name",
    type=click.STRING,
    default=create_default_output_name(),
    show_default=True,
    help="The resulting system evaluation is written to the parent directory with this name.",
)
@run_options()
@click.option(
    "--docker-cleanup",
    type=click.BOOL,
    default=False,
    show_default=True,
    help=(
        "If given, images and containers will be removed after each use to avoid disk pressure. "
        "Enabling this will make things slower but consume less disk."
    ),
)
@click.option(
    "--force-rebuild",
    type=click.BOOL,
    default=False,
    show_default=True,
    help=(
        "If given, the underlying docker containers are rebuilt from scratch before running. "
        "This will take more time that using the existing images."
    ),
)
@click.option(
    "--downloads",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True, path_type=Path),
    show_default=True,
    help=(
        "Some benchmark docker containers need to download test assets. If no path is given here, "
        "those assets are downloaded into the image. If this is given, the assets are downloaded "
        "to this location which is bind mounted into the container so the downloads can be re-used."
    ),
)
def benchmark(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    test: Tuple[BenchmarkName, ...],
    gpu: Tuple[GPUIdentity, ...],
    output_name: str,
    output_parent: Path,
    title: str,
    description: str,
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads: Optional[Path],
) -> None:
    """
    Run one or more benchmarks and records the results.

    \f

    :param test: See click help for docs!
    :param gpu: See click help for docs!
    :param output_name: See click help for docs!
    :param output_parent: See click help for docs!
    :param title: See click help for docs!
    :param description: See click help for docs!
    :param docker_cleanup: See click help for docs!
    :param force_rebuild: See click help for docs!
    :param downloads: See click help for docs!
    :return: None
    """

    start_time = datetime.now()

    if not test:
        test = tuple(test for test in BenchmarkName)

    gpu_discovery_output = discover_gpus()

    if not gpu:
        gpu = gpu_discovery_output

    # Read the system info first. If there are problems we want errors before running any tests.
    cpu = discover_cpu()

    if gpu != gpu_discovery_output and output_name != create_default_output_name():
        # Specific set of GPUs are being used, but the output name has not been re-set by user.
        # need to re-do name creation.
        output_path = output_parent / create_default_output_name(gpus=gpu, cpu=cpu)
    else:
        # Base case, output name will match the actual hardware setup, or it has been overwritten.
        output_path = output_parent / output_name

    mem_info = psutil.virtual_memory()

    total_memory_bytes = mem_info.total

    # Create the tests the user requested and run them.

    results: List[BenchmarkResult] = []
    failed_tests: List[BenchmarkName] = []

    for index, named_executor in enumerate(
        render_named_executors(
            benchmarks=test,
            gpu=gpu,
            docker_cleanup=docker_cleanup,
            force_rebuild=force_rebuild,
            downloads_directory_path=downloads,
        ),
        start=1,
    ):
        if named_executor is not None:
            LOGGER.info(
                f"Executing benchmark {index}/{len(test)}: {named_executor.benchmark_name} ..."
            )
            try:
                result = named_executor.executor()
                results.append(result)
                result_number: float = round(
                    getattr(result.numerical_results, result.critical_result_key), 2
                )

                result_log = f"Result: {result_number} {result.unit}"

                if result.gpu_power_usage is not None:
                    critical_power_value: float = round(
                        getattr(result.gpu_power_usage, result.critical_gpu_power_usage_key), 2
                    )
                    result_log = ", ".join(
                        [
                            result_log,
                            f"{critical_power_value} Joules",
                        ]
                    )

                LOGGER.info(result_log)

            except Exception as e:  # pylint: disable=broad-except
                failed_tests.append(named_executor.benchmark_name)
                pretty_exception = re.sub(r"\s+", " ", str(e))[:1000] + "..."
                LOGGER.info(f"Failed! Err: {pretty_exception}")

    # Tests are complete, write the output.

    LOGGER.info("Benchmarking Complete!")

    system_evaluation = SystemEvaluation(
        title=title,
        description=description,
        gpu_box_benchmark_version=_GPU_BOX_BENCHMARK_VERSION,
        cpu=cpu,
        total_memory_gb=round(total_memory_bytes / (1024**3), 3),
        gpus=gpu,
        results=results,
        start_time=start_time,
        runtime_seconds=(datetime.now() - start_time).total_seconds(),
        failed_tests=failed_tests,
    )

    evaluation_string = system_evaluation.model_dump_json(indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(evaluation_string)

    LOGGER.info(f"Output written to {output_path}")
    LOGGER.info(f"System Evaluation: {evaluation_string}")


@cli.command(short_help="Prints a description about what each of the supported benchmarks do.")
@click.option(
    "--width",
    default=100,
    show_default=True,
    help="Printout Width",
    type=click.IntRange(min=10),
)
def explain_benchmarks(
    width: int,
) -> None:  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """
    To try to keep the main benchmark command clean, this command describes each of the included
    benchmarks and their variants.

    \f

    :param width: See click help for docs!
    :return: None
    """

    for family in BenchmarkFamily:
        tests = " ".join(
            [
                f"{test.value} (Pretty: {BENCHMARK_TO_PRETTY[test]})"
                for test in BenchmarkName
                if family.name in test.value
            ]
        )
        click.echo(
            textwrap.fill(f"Benchmark Family: {family.name}, Tests: {tests}", width=width) + "\n"
        )
        click.echo(f"{textwrap.fill(EXTENDED_BENCHMARK_DOCUMENTS[family], width=width)} \n\n")


@cli.command(short_help="Creates a systemd unit that will execute a benchmarking run at boot.")
@click.option(
    "--output-path",
    default=Path("./gpu_box_benchmark.service").resolve(),
    show_default=True,
    help="The resulting systemd service will be written to this path.",
    type=click.Path(
        file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
)
@run_options()
def render_systemd(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    output_path: Path,
    title: str,
    description: str,
    output_parent: Path,
) -> None:
    """
    Creates a systemd unit that will start a benchmark at boot. The systemd unit only has the
    output parent, title and description filled via the environment. All other parameters are left
    as their default meaning the command will run all tests on all attached GPUs. Output parent
    in this case can be a NAS device.

    \f

    :param output_path: See click help for docs!
    :param title: See click help for docs!
    :param description: See click help for docs!
    :param output_parent: See click help for docs!
    :return: None
    """

    file_contents = render_systemd_file(
        output_parent=output_parent,
        title=title,
        description=description,
        path_to_python_file=os.path.abspath(__file__),
    )

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(file_contents)

    click.echo(f"Wrote systemd unit to {output_path}")
    click.echo(file_contents)


INPUT_OPTION = click.option(
    "--input",
    "input_path",
    help="The set of benchmark JSON files to compare. Can be given multiple times",
    type=click.Path(
        file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    multiple=True,
)

LEGEND_LABEL_OPTION = options.create_enum_option(
    "--legend-label",
    help_message=(
        "Default basis for the legend label. This will be overriden if a label is "
        "specifically provided for the run."
    ),
    input_enum=LabelBase,
    default=LabelBase.gpu,
)

LABELED_INPUT_OPTION = click.option(
    "--labeled-input",
    help="A benchmark JSON file and its label. Can be given multiple times.",
    type=(
        click.Path(
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        str,
    ),
    multiple=True,
)

OUTPUT_OPTION = click.option(
    "--output-path",
    default=Path("./comparison.png").resolve(),
    show_default=True,
    help="The resulting visualization will be written to this path.",
    type=click.Path(
        file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
)

TITLE_OPTION = click.option(
    "--title",
    default=None,
    show_default=True,
    help="Written on the top of the figure.",
    type=click.STRING,
)

WATERMARK_OPTION = click.option(
    "--watermark",
    required=False,
    help="Subtly written along the right side of the image.",
    type=click.STRING,
)

FONT_FAMILY_OPTION = click.option(
    "--font-family",
    required=False,
    help="Passed to rcParams['font.family'].",
    type=click.STRING,
)


@cli.command(name="compare-systems", short_help="Draw graphs comparing benchmark outputs.")
@options.create_enum_option(
    "--mode",
    help_message=(
        "Decides which data should be plotted within the input files to compare the systems."
    ),
    input_enum=SystemComparisonPlotMode,
    default=SystemComparisonPlotMode.results,
    multiple=False,
)
@INPUT_OPTION
@LEGEND_LABEL_OPTION
@LABELED_INPUT_OPTION
@OUTPUT_OPTION
@TITLE_OPTION
@click.option(
    "--flatten-versions",
    default=True,
    show_default=True,
    help="If given tests of different versions will be combined in the result.",
    type=click.BOOL,
)
@options.create_enum_option(
    "--only-test",
    help_message="Only include this test / these tests in the output visualization.",
    input_enum=BenchmarkName,
    multiple=True,
)
@click.option(
    "--n-cols",
    default=4,
    show_default=True,
    help="If given tests of different versions will be combined in the result.",
    type=click.IntRange(min=1),
)
@WATERMARK_OPTION
@click.option(
    "--y-axis-pad-pct",
    required=False,
    help="Numerical pad percent applied to the Y-Axes of the subplots.",
    type=click.FloatRange(min=0),
    default=0.1,
)
@FONT_FAMILY_OPTION
def compare_systems(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    mode: SystemComparisonPlotMode,
    input_path: Tuple[Path, ...],
    labeled_input: Tuple[Tuple[Path, str], ...],
    legend_label: LabelBase,
    output_path: Path,
    title: Optional[str],
    flatten_versions: bool,
    only_test: Tuple[BenchmarkName, ...],
    n_cols: int,
    watermark: Optional[str],
    y_axis_pad_pct: Optional[float],
    font_family: Optional[str],
) -> None:
    """
    Draws a matplotlib visualization comparing the results of the different input benchmarks.

    \f

    :param mode: See click help for docs!
    :param input_path: See click help for docs!
    :param labeled_input: See click help for docs!
    :param legend_label: See click help for docs!
    :param output_path: See click help for docs!
    :param title: See click help for docs!
    :param flatten_versions: See click help for docs!
    :param only_test: See click help for docs!
    :param n_cols: See click help for docs!
    :param watermark: See click help for docs!
    :param y_axis_pad_pct: See click help for docs!
    :param font_family: See click help for docs!
    :return: None
    """

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    plot_function = (
        render_plot.create_system_comparison_visualization_results
        if mode == SystemComparisonPlotMode.results
        else render_plot.create_system_comparison_visualization_results_power
    )

    plot_function(
        evaluations_and_base=EvaluationsBase(
            evaluations=tuple(map(load_system_evaluation_from_disk, input_path)),
            label_base=legend_label,
        ),
        labeled_evaluations=tuple(
            (
                (load_system_evaluation_from_disk(path=evaluation_path), label)
                for evaluation_path, label in labeled_input
            )
        ),
        flatten_versions=flatten_versions,
        output_path=output_path,
        title=title if title is not None else "Comparison",
        only_render_tests=only_test,
        n_cols=n_cols,
        watermark=watermark,
        y_axis_pad_pct=y_axis_pad_pct,
    )


def cli_inputs_to_llm_results(
    input_paths: Tuple[Path, ...],
    legend_label: LabelBase,
    labeled_input: Tuple[Tuple[Path, str], ...],
) -> Tuple[LLMThroughputResult, ...]:
    """
    Helper function to take user input and generate a set of NTs for plotting.
    :param input_paths: List of input evaluations on disk.
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

    return results


@cli.command(
    name="compare-llms",
    short_help="Draw graphs comparing LLM context depth, Quantization and System Performance.",
)
@INPUT_OPTION
@LEGEND_LABEL_OPTION
@LABELED_INPUT_OPTION
@OUTPUT_OPTION
@TITLE_OPTION
@WATERMARK_OPTION
@FONT_FAMILY_OPTION
@options.create_enum_option(
    "--llm",
    help_message="Decides which benchmark to run.",
    input_enum=LLMName,
    default=LLMName.qwen_2_5_14b_instruct,
    multiple=False,
)
def compare_llms(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    input_path: Tuple[Path, ...],
    legend_label: LabelBase,
    labeled_input: Tuple[Tuple[Path, str], ...],
    output_path: Path,
    title: Optional[str],
    watermark: Optional[str],
    font_family: Optional[str],
    llm: LLMName,
) -> None:
    """
    Draws a matplotlib visualization comparing the results of the different input benchmarks.

    \f

    :param input_path: See click help for docs!
    :param legend_label: See click help for docs!
    :param labeled_input: See click help for docs!
    :param output_path: See click help for docs!
    :param title: See click help for docs!
    :param watermark: See click help for docs!
    :param font_family: See click help for docs!
    :param llm: See click help for docs!
    :return: None
    """

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    results: Tuple[LLMThroughputResult, ...] = cli_inputs_to_llm_results(
        input_paths=input_path,
        legend_label=legend_label,
        labeled_input=labeled_input,
    )

    render_plot.create_llm_throughput_visualization(
        results=results,
        plot_model=llm,
        output_path=output_path,
        title=title if title is not None else llm.value,
        watermark=watermark,
    )


@cli.command(
    name="compare-llm-families",
    short_help=(
        "Draw graphs comparing LLM context depth, Quantization and "
        "System Performance for all models families."
    ),
)
@INPUT_OPTION
@LEGEND_LABEL_OPTION
@LABELED_INPUT_OPTION
@click.option(
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False, writable=True, path_type=Path),
    default=Path().cwd().joinpath("llm_families"),
    show_default=True,
    help="Output viz are all written to this directory.",
)
@WATERMARK_OPTION
@FONT_FAMILY_OPTION
def compare_llm_families(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    input_path: Tuple[Path, ...],
    legend_label: LabelBase,
    labeled_input: Tuple[Tuple[Path, str], ...],
    output_dir: str,
    watermark: Optional[str],
    font_family: Optional[str],
) -> None:
    """
    Iterates through all supported LLMs and generates a comparison visualization for each.

    \f

    :param input_path: See click help for docs!
    :param legend_label: See click help for docs!
    :param labeled_input: See click help for docs!
    :param output_dir: Directory to save the generated images.
    :param watermark: See click help for docs!
    :param font_family: See click help for docs!
    :return: None
    """

    if font_family is not None:
        plt.rcParams["font.family"] = font_family

    results: Tuple[LLMThroughputResult, ...] = cli_inputs_to_llm_results(
        input_paths=input_path,
        legend_label=legend_label,
        labeled_input=labeled_input,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all supported LLMs and generate plots
    for index, model in enumerate(LLMName):

        model = LLMName(model)

        sanitized_model = model.value.replace(" ", "_").lower()
        output_path = Path(output_dir) / f"llm_{index}_{sanitized_model}.png"

        render_plot.create_llm_throughput_visualization(
            results=results,
            plot_model=model,
            output_path=output_path,
            title=model.value,
            watermark=watermark,
        )


if __name__ == "__main__":
    cli()
