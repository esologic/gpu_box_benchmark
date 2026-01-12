"""Main module."""

import logging
from pathlib import Path
from typing import List, Tuple

import click
from bonus_click import options
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup

from gpu_box_benchmark import nvidia_deep_learning_examples_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkName, CreateBenchmarkExecutor
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

_GPU_BOX_BENCHMARK_VERSION = "1.0.0"


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
)
@optgroup.group(
    "GPU Configuration", help="Decide which GPUs to use.", cls=RequiredMutuallyExclusiveOptionGroup
)
@optgroup.option(
    "--gpu",
    "-g",
    type=click.Choice(choices=discover_gpus()),
    help="The GPU(s) to use for computation. Can be given multiple times.",
    required=False,
    multiple=True,
)
@optgroup.option(
    "--all-gpus",
    type=click.BOOL,
    is_flag=True,
    help="Use all GPUs attached to the system",
    required=False,
    default=True,
)
@click.option(
    "--output-path",
    type=click.Path(exists=False, writable=True, file_okay=True, dir_okay=False, path_type=Path),
    default=Path("./benchmark_result.json").resolve(),
    show_default=True,
    help="Output path",
)
def benchmark(
    test: BenchmarkName, gpu: Tuple[GPUDescription, ...], all_gpus: bool, output_path: Path
) -> None:  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """
    Run one or more benchmarks and records the results.

    \f

    TODO: We may want to check how loaded down the system is before starting benchmarks.

    :return: None
    """

    if all_gpus:
        gpu = discover_gpus()

    creation_functions: List[CreateBenchmarkExecutor] = [
        nvidia_deep_learning_examples_wrapper.deep_learning_examples_lookup
    ]

    requested_tests = [test]

    executors = [
        next(
            filter(
                None,
                (
                    creation_function(benchmark_name=requested_test, gpus=gpu)
                    for creation_function in creation_functions
                ),
            )
        )
        for requested_test in requested_tests
    ]

    results: List[NumericalBenchmarkResult] = [executor() for executor in executors]

    system_evaluation = SystemEvaluation(
        title="Title",
        description="Description",
        gpu_box_benchmark_version=_GPU_BOX_BENCHMARK_VERSION,
        cpu_name="CPU Name",
        cpu_count=1,
        memory_mb=32000,
        gpus=gpu,
        results=results,
    )

    evaluation_string = system_evaluation.model_dump_json()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(evaluation_string)

    print(evaluation_string)


@cli.command(short_help="Prints a description about what each of the supported benchmarks do.")
def explain_benchmarks() -> (  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    None
):
    """
    To try to keep the main benchmark command clean, this command describes each of the included
    benchmarks and their variants.

    \f

    :return: None
    """


if __name__ == "__main__":
    cli()
