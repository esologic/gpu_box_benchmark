"""
Code for running the benchmarking tool in content aware timelapse and parsing the output.
"""

import json
import logging
from typing import List, Optional, Tuple

from benchmark_dockerfiles import CONTENT_AWARE_TIMELAPSE_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult

LOGGER = logging.getLogger(__name__)

_CONTENT_AWARE_TIMELAPSE_BENCHMARK_VERSION = "0.1.0"


def _parse_fps(container_outputs: ContainerOutputs) -> float:
    """
    Finds and returns the audio frames / second score.
    :param container_outputs: Contains the full output from the docker container post run.
    :return: Frames / Second as a float.
    """
    prefix = "Benchmark Result: "
    result_line = next(
        iter([line for line in container_outputs.logs.splitlines() if prefix in line])
    )
    loaded = json.loads(result_line.replace(prefix, ""))

    return float(loaded["throughput_fps"])


def create_content_aware_timelapse_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run the benchmark in content aware timelapse.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    if benchmark_name == BenchmarkName.content_aware_timelapse_vit_scores:
        envs: List[Tuple[str, str]] = [("RUN_ENV", "--backend-scores vit-cls")]
    elif benchmark_name == BenchmarkName.content_aware_timelapse_vit_attention:
        envs = [("RUN_ENV", "--backend-scores vit-attention")]
    else:
        return None

    def run_content_aware_timelapse_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = True

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_CONTENT_AWARE_TIMELAPSE_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Frames / Second",
            unit="f/s",
            multi_gpu_native=multi_gpu_native,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=CONTENT_AWARE_TIMELAPSE_DOCKERFILE,
                tag=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: envs,
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_fps,
            ),
        )

    return run_content_aware_timelapse_docker
