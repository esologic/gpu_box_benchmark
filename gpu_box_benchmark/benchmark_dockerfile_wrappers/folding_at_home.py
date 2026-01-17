"""
Code for running the FAHBench and parsing the output.
Doesn't yet support multiple GPUs.
"""

# pylint: disable=duplicate-code

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from benchmark_dockerfiles import FAHBENCH_BENCHMARK_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import build_run_dockerfile_read_logs
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

    if len(gpus) > 1:
        LOGGER.debug("Multi-GPU not yet supported in the FAHBench benchmark.")
        return None

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

        final_scores = [
            _parse_final_score(
                build_run_dockerfile_read_logs(
                    dockerfile_path=FAHBENCH_BENCHMARK_DOCKERFILE,
                    tag=benchmark_name.value,
                    gpus=gpus,
                    env_vars=envs,
                )
            )
            for _ in range(_RUNS_PER_BENCHMARK)
        ]

        summary_dict = pd.Series(final_scores).dropna().describe().to_dict()

        results = ReportFileNumerical(
            sample_count=summary_dict["count"],
            mean=summary_dict["mean"],
            std=summary_dict["std"],
            result_min=summary_dict["min"],
            percentile_25=summary_dict["25%"],
            percentile_50=summary_dict["50%"],
            percentile_75=summary_dict["75%"],
            result_max=summary_dict["max"],
        )

        return NumericalBenchmarkResult(
            percentile_50=results.percentile_50,
            percentile_75=results.percentile_75,
            result_max=results.result_max,
            name=benchmark_name.value,
            benchmark_version=_FAH_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Nanoseconds / Day",
            unit="ns/day",
            sample_count=results.sample_count,
            std=results.std,
            mean=results.mean,
            result_min=results.result_min,
            percentile_25=results.percentile_25,
        )

    return run_fah_bench_docker
