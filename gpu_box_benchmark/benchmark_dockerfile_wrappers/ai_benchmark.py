"""
Code for running the ai-benchmark and parsing the output.
Doesn't yet support multiple GPUs.
"""

# pylint: disable=duplicate-code

import logging
from typing import Optional, Tuple

import pandas as pd

from benchmark_dockerfiles import AI_BENCHMARK_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import build_run_dockerfile_read_logs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult

LOGGER = logging.getLogger(__name__)

_AI_BENCHMARK_VERSION = "0.1.0"


def _parse_ai_score(docker_logs: str) -> float:
    """
    Finds and returns the ai score from some ai-benchmark logs.
    :param docker_logs: Full output from the docker container post run. Should be a single line.
    :return: AI Score.
    """

    return float(next(iter(docker_logs.splitlines())))


def create_ai_benchmark_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run the ai-benchmark suite.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    if len(gpus) > 1:
        LOGGER.debug("Multi-GPU not yet supported in the FAHBench benchmark.")
        return None

    if benchmark_name != BenchmarkName.ai_benchmark:
        return None

    def run_ai_benchmark_docker() -> NumericalBenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        final_score = _parse_ai_score(
            build_run_dockerfile_read_logs(
                dockerfile_path=AI_BENCHMARK_DOCKERFILE,
                tag=benchmark_name.value,
                gpus=gpus,
                env_vars=[],
            )
        )

        return NumericalBenchmarkResult(
            percentile_50=final_score,
            percentile_75=final_score,
            result_max=final_score,
            name=benchmark_name.value,
            benchmark_version=_AI_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="AI Score",
            unit="pts",
            sample_count=1,
            std=0,
            mean=final_score,
            result_min=final_score,
            percentile_25=final_score,
        )

    return run_ai_benchmark_docker
