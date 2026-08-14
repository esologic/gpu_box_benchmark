"""Main module."""

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

from gpu_box_benchmark import visualization
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
from gpu_box_benchmark.visualization import EvaluationsBase, XLabelBase

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
                pretty_exception = re.sub(r"\s+", " ", str(e))[:100] + "..."
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


@cli.command(short_help="Draws graphs comparing benchmark outputs.")
@click.option(
    "--input",
    "input_path",
    help="The set of benchmark JSON files to compare. Can be given multiple times",
    type=click.Path(
        file_okay=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    multiple=True,
)
@options.create_enum_option(
    "--x-label",
    help_message=(
        "Default basis for the X label. This will be overriden if a label is "
        "specifically provided for the run."
    ),
    input_enum=XLabelBase,
    default=XLabelBase.gpu,
)
@click.option(
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
@click.option(
    "--output-path",
    default=Path("./comparison.png").resolve(),
    show_default=True,
    help="The resulting visualization will be written to this path.",
    type=click.Path(
        file_okay=True, dir_okay=False, writable=True, resolve_path=True, path_type=Path
    ),
)
@click.option(
    "--title",
    default="Comparison",
    show_default=True,
    help="Written on the top of the figure.",
    type=click.STRING,
)
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
@click.option(
    "--watermark",
    required=False,
    help="Subtly written along the right side of the image.",
    type=click.STRING,
)
@click.option(
    "--y-axis-pad-pct",
    required=False,
    help="Numerical pad percent applied to the Y-Axes of the subplots.",
    type=click.FloatRange(min=0),
)
@click.option(
    "--font-family",
    required=False,
    help="Passed to rcParams['font.family'].",
    type=click.STRING,
)
def compare(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    input_path: Tuple[Path, ...],
    labeled_input: Tuple[Tuple[Path, str], ...],
    x_label: XLabelBase,
    output_path: Path,
    title: str,
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

    :param input_path: See click help for docs!
    :param labeled_input: See click help for docs!
    :param x_label: See click help for docs!
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

    visualization.create_comparison_visualization(
        evaluations_and_base=EvaluationsBase(
            evaluations=tuple(map(load_system_evaluation_from_disk, input_path)),
            x_label_base=x_label,
        ),
        labeled_evaluations=tuple(
            (
                (load_system_evaluation_from_disk(path=evaluation_path), label)
                for evaluation_path, label in labeled_input
            )
        ),
        flatten_versions=flatten_versions,
        output_path=output_path,
        title=title,
        only_render_tests=only_test,
        n_cols=n_cols,
        watermark=watermark,
        y_axis_pad_pct=y_axis_pad_pct,
    )


if __name__ == "__main__":
    cli()
