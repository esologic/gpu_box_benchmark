"""Main module."""

import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple

import click
import psutil
from bonus_click import options
from click.decorators import FC

from gpu_box_benchmark import visualization
from gpu_box_benchmark.benchmark_dockerfile_wrappers import (
    ai_benchmark,
    blender_benchmark,
    content_aware_timelapse,
    folding_at_home,
    hashcat,
    llama_bench,
    nvidia_deep_learning_examples,
    nvidia_gds,
    whisper,
)
from gpu_box_benchmark.benchmark_jobs import (
    EXTENDED_BENCHMARK_DOCUMENTS,
    BenchmarkExecutor,
    BenchmarkFamily,
    BenchmarkName,
    CreateBenchmarkExecutor,
)
from gpu_box_benchmark.cli_common import (
    BENCHMARK_COMMAND_NAME,
    ENV_VAR_MAPPING,
    create_default_output_name,
)
from gpu_box_benchmark.locate_describe_hardware import (
    GPU_CLICK_OPTION,
    GPUIdentity,
    discover_cpu,
    discover_gpus,
)
from gpu_box_benchmark.numeric_benchmark_result import (
    BenchmarkResult,
    SystemEvaluation,
    load_system_evaluation_from_disk,
)
from gpu_box_benchmark.render_systemd import render_systemd_file

LOGGER_FORMAT = "[%(asctime)s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)

_GPU_BOX_BENCHMARK_VERSION = "0.1.0"
"""
See top level `CHANGELOG.md` for version history of this format.
"""


class NamedExecutor(NamedTuple):
    """
    Intermediate Type for labeling executors.
    """

    benchmark_name: BenchmarkName
    executor: BenchmarkExecutor


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
def benchmark(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    test: Tuple[BenchmarkName, ...],
    gpu: Tuple[GPUIdentity, ...],
    output_name: str,
    output_parent: Path,
    title: str,
    description: str,
    docker_cleanup: bool,
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

    creation_functions: List[CreateBenchmarkExecutor] = [
        nvidia_deep_learning_examples.create_resnet50_executor,
        llama_bench.create_llama_bench_executor,
        blender_benchmark.create_blender_benchmark_executor,
        folding_at_home.create_fah_bench_executor,
        ai_benchmark.create_ai_benchmark_executor,
        whisper.create_whisper_executor,
        content_aware_timelapse.create_content_aware_timelapse_executor,
        nvidia_gds.create_gdsio_executor,
        hashcat.create_hashcat_executor,
    ]

    def executor_for_test(requested_test: BenchmarkName) -> BenchmarkExecutor:
        """
        Look up the executor for the input test. Adds logging.
        :param requested_test: From CLI.
        :return: Executor for the test.
        """
        try:
            return next(
                filter(
                    None,
                    (
                        creation_function(
                            benchmark_name=requested_test,
                            gpus=gpu,
                            docker_cleanup=docker_cleanup,
                        )
                        for creation_function in creation_functions
                    ),
                ),
            )
        except StopIteration as lookup_error:
            raise StopIteration(
                f"Couldn't find executor for test: {requested_test.value}"
            ) from lookup_error

    try:
        named_executors: List[Optional[NamedExecutor]] = [
            NamedExecutor(
                benchmark_name=requested_test,
                executor=executor_for_test(requested_test=requested_test),
            )
            for requested_test in test
        ]
    except StopIteration as e:
        raise ValueError("No valid tests for the given input.") from e

    results: List[BenchmarkResult] = []

    for named_executor in named_executors:
        if named_executor is not None:
            LOGGER.info(f"Executing benchmark: {named_executor.benchmark_name} ...")
            try:
                result = named_executor.executor()
                results.append(result)
                result_number: float = round(
                    getattr(result.numerical_results, result.critical_result_key), 2
                )
                LOGGER.info(f"Result: {result_number} {result.unit}")
            except Exception as _e:  # pylint: disable=broad-except
                LOGGER.exception(f"Benchmark: {named_executor.benchmark_name} failed!")

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
        tests = " ".join([test.value for test in BenchmarkName if family.name in test.value])
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
def compare(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    input_path: Tuple[Path, ...],
    output_path: Path,
    title: str,
) -> None:
    """
    Draws a matplotlib visualization comparing the results of the different input benchmarks.

    \f

    :param input_path: See click help for docs!
    :param output_path: See click help for docs!
    :param title: See click help for docs!
    :return: None
    """

    visualization.create_comparison_visualization(
        evaluations=tuple(map(load_system_evaluation_from_disk, input_path)),
        output_path=output_path,
        title=title,
        n_cols=4,
        max_bar_label_length=15,
    )


if __name__ == "__main__":
    cli()
