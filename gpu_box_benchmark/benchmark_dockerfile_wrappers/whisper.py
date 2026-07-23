"""
Code for running whisper as a benchmark.
Whisper doesn't support having multiple GPUs work through the same audio file, so this benchmark
doesn't natively support multiple GPUs.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from benchmark_dockerfiles import WHISPER_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_types import (
    BenchmarkExecutor,
    BenchmarkName,
    BenchmarkResult,
    NumericalResultKey,
)
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity

LOGGER = logging.getLogger(__name__)

_WHISPER_BENCHMARK_VERSION = "0.2.0"
"""
# Version History

## 0.2.0 - (2026-02-03)
* Switched to base image w/CUDA 11.4.3 to support Kepler era cards.

## 0.1.0 - (2026-01-20)
* First version 
"""


def _parse_frames_per_second(container_outputs: ContainerOutputs) -> float:
    """
    Finds and returns the processing speed of the input audio.
    :param container_outputs: All outputs from the container, contains the logs which are the full
    output from the docker container post run.
    :return: Final score.
    """

    return float(next(iter(container_outputs.logs.splitlines())))


def create_whisper_executor(  # pylin
    benchmark_name: BenchmarkName,
    gpus: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to transcribe audio with whisper and records the output.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :param force_rebuild: If given, the underlying docker containers are rebuilt from scratch
    before running. This will take more time that using the existing images.
    :param downloads_directory_path: Benchmark docker containers may need to download test assets
    during build time or runtime. By default, if this is not given, those assets are downloaded
    into the image. If this is given, the assets are downloaded to this location which is bind
    mounted into the container so the downloads can be re-used.
    :return: The callable to run the benchmark.
    """

    if benchmark_name != BenchmarkName.whisper_medium_fp16:
        return None

    def run_whisper_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = False

        return BenchmarkResult(
            name=benchmark_name,
            benchmark_version=_WHISPER_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Audio Frames / Second",
            unit="f/s",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.forced_multi_gpu_sum,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=WHISPER_DOCKERFILE,
                benchmark_name=benchmark_name.value,
                benchmark_version=_WHISPER_BENCHMARK_VERSION,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: [],
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_frames_per_second,
                docker_cleanup=docker_cleanup,
                force_rebuild=force_rebuild,
                downloads_directory_path=downloads_directory_path,
            ),
        )

    return run_whisper_docker
