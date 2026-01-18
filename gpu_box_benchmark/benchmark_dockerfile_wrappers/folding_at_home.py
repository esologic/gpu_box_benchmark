"""
Code for running the FAHBench and parsing the output.
Doesn't yet support multiple GPUs.
"""

# pylint: disable=duplicate-code

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from benchmark_dockerfiles import FAHBENCH_BENCHMARK_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

LOGGER = logging.getLogger(__name__)

_FAH_BENCHMARK_VERSION = "0.1.0"
_RUNS_PER_BENCHMARK = 3


def _parse_final_score(docker_logs: str) -> float:
    """
    Finds and returns the final score from some FAHBench logs.
    :param docker_logs: Full output from the docker container post run.
    :return: Final score.
    """

    lines = docker_logs.splitlines()

    final_score_line = next(iter([line for line in lines if "Final score" in line]))

    final_score = float(final_score_line.split(":")[1].strip())

    return final_score


def create_fah_bench_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some Folding@Home (FAHBench) benchmarks.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    name_to_parameters: Dict[BenchmarkName, List[Tuple[str, Union[str, float, bool, int]]]] = {
        BenchmarkName.fah_bench_single: [
            ("FAHBENCH_PRECISION", "single"),
        ],
        BenchmarkName.fah_bench_double: [
            ("FAHBENCH_PRECISION", "double"),
        ],
    }

    envs: Optional[List[Tuple[str, Union[str, float, bool, int]]]] = name_to_parameters.get(
        benchmark_name, None
    )

    if envs is None:
        return None

    def run_fah_bench_docker() -> NumericalBenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = False

        results = docker_wrapper.benchmark_dockerfile(
            dockerfile_path=FAHBENCH_BENCHMARK_DOCKERFILE,
            tag=benchmark_name.value,
            gpus=gpus,
            env_vars=envs,
            multi_gpu_native=multi_gpu_native,
            logs_to_result=_parse_final_score,
        )

        return NumericalBenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_FAH_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Nanoseconds / Day",
            unit="ns/day",
            multi_gpu_native=multi_gpu_native,
            min_by_gpu_type=results.min_by_gpu_type,
            max_by_gpu_type=results.max_by_gpu_type,
            mean_by_gpu_type=results.mean_by_gpu_type,
            theoretical_sum=results.theoretical_sum,
            parallel_mean=results.parallel_mean,
            experimental_sum=results.experimental_sum,
        )

    return run_fah_bench_docker
