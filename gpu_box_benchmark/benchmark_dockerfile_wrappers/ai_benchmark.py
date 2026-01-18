"""
Code for running the ai-benchmark and parsing the output.
Doesn't yet support multiple GPUs.
"""

# pylint: disable=duplicate-code

import logging
from typing import Optional, Tuple

from benchmark_dockerfiles import AI_BENCHMARK_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult

LOGGER = logging.getLogger(__name__)

_AI_BENCHMARK_VERSION = "0.1.0"


def _parse_ai_score(container_outputs: ContainerOutputs) -> float:
    """
    Finds and returns the ai score from some ai-benchmark logs.
    :param docker_logs: Full output from the docker container post run. Should be a single line.
    :return: AI Score.
    """

    return float(next(iter(container_outputs.logs.splitlines())))


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

    if benchmark_name != BenchmarkName.ai_benchmark:
        return None

    def run_ai_benchmark_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = True

        results = docker_wrapper.benchmark_dockerfile(
            dockerfile_path=AI_BENCHMARK_DOCKERFILE,
            tag=benchmark_name.value,
            gpus=gpus,
            create_runtime_env_vars=lambda runtime_gpus: [],
            multi_gpu_native=multi_gpu_native,
            outputs_to_result=_parse_ai_score,
        )

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_AI_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="AI Score",
            unit="pts",
            numerical_results=results,
        )

    return run_ai_benchmark_docker
