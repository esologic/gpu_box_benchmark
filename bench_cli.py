"""Main module."""

import logging
from pathlib import Path
from typing import List, NamedTuple, Tuple

import click
import cpuinfo
import psutil
from bonus_click import options

from gpu_box_benchmark import nvidia_deep_learning_examples_wrapper
from gpu_box_benchmark.benchmark_jobs import (
    BenchmarkExecutor,
    BenchmarkName,
    CreateBenchmarkExecutor,
)
from gpu_box_benchmark.gpu_discovery import GPUDescription, discover_gpus
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, SystemEvaluation

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)

_GPU_BOX_BENCHMARK_VERSION = "0.1.0"


class NamedExecutor(NamedTuple):
    """
    Intermediate Type for labeling executors.
    """

    benchmark_name: BenchmarkName
    executor: BenchmarkExecutor


@click.group()
def cli() -> None:
    """
    Uses Docker to run GPU related benchmarks with the goal of comparing system hardware
    architectures.

    \f

    :return: None
    """


@cli.command(short_help="Run one or more benchmarks and records the results.")
@options.create_enum_option(
    "--test",
    help_message="Decides which benchmark to run.",
    default=BenchmarkName.resnet50_infer_batch_256_amp,
    input_enum=BenchmarkName,
    multiple=True,
)
@click.option(
    "--gpu",
    "-g",
    type=click.Choice(choices=discover_gpus()),
    help=(
        "The GPU(s) to use for computation. Can be given multiple times. "
        "If not given, all GPUs will be used."
    ),
    multiple=True,
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, writable=True, file_okay=True, dir_okay=False, path_type=Path),
    default=Path("./benchmark_result.json").resolve(),
    show_default=True,
    help="The resulting system evaluation is written to this path.",
)
@click.option(
    "--title",
    type=click.STRING,
    help="A short few words to describe the run. Will be a plot title in resulting visualizations.",
    default="Sample Title",
    show_default=True,
)
@click.option(
    "--description",
    type=click.STRING,
    help="Longer text field to qualitatively describe the run in a more verbose way.",
    default="Sample Description. This run was completed on a computer made of corn!",
    show_default=True,
)
def benchmark(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    test: Tuple[BenchmarkName, ...],
    gpu: Tuple[GPUDescription, ...],
    output_path: Path,
    title: str,
    description: str,
) -> None:
    """
    Run one or more benchmarks and records the results.

    :param test: See click help for docs!
    :param gpu: See click help for docs!
    :param output_path: See click help for docs!
    :param title: See click help for docs!
    :param description: See click help for docs!
    :return: None

    \f

    TODO: We may want to check how loaded down the system is before starting benchmarks.

    :return: None
    """

    if not gpu:
        gpu = discover_gpus()

    # Read the system info first. If there are problems we want errors before running any tests.

    cpu_info = cpuinfo.get_cpu_info()
    mem_info = psutil.virtual_memory()

    cpu_name = cpu_info.get("brand_raw", "Unknown")
    physical_cpus = psutil.cpu_count(logical=False)
    logical_cpus = psutil.cpu_count(logical=True)
    total_memory_bytes = mem_info.total

    # Create the tests the user requested and run them.

    creation_functions: List[CreateBenchmarkExecutor] = [
        nvidia_deep_learning_examples_wrapper.create_resnet50_executor
    ]

    named_executors: List[NamedExecutor] = [
        NamedExecutor(
            benchmark_name=requested_test,
            executor=next(
                filter(
                    None,
                    (
                        creation_function(benchmark_name=requested_test, gpus=gpu)
                        for creation_function in creation_functions
                    ),
                )
            ),
        )
        for requested_test in test
    ]

    results: List[NumericalBenchmarkResult] = []

    for named_executor in named_executors:
        LOGGER.info(f"Executing benchmark: {named_executor.benchmark_name} ...")
        results.append(named_executor.executor())

    # Tests are complete, write the output.

    LOGGER.info("Benchmarking Complete!")

    system_evaluation = SystemEvaluation(
        title=title,
        description=description,
        gpu_box_benchmark_version=_GPU_BOX_BENCHMARK_VERSION,
        cpu_name=cpu_name,
        physical_cpus=physical_cpus,
        logical_cpus=logical_cpus,
        total_memory_gb=total_memory_bytes / (1024**3),
        gpus=gpu,
        results=results,
    )

    evaluation_string = system_evaluation.model_dump_json(indent=2)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(evaluation_string)

    LOGGER.info(f"Output written to {output_path}")
    LOGGER.info(f"System Evaluation: {evaluation_string}")


@cli.command(short_help="Prints a description about what each of the supported benchmarks do.")
def explain_benchmarks() -> (
    None
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """
    To try to keep the main benchmark command clean, this command describes each of the included
    benchmarks and their variants.

    \f

    :return: None
    """


if __name__ == "__main__":
    cli()
