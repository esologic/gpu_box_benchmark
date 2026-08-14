"""
Code for running the "nvidia deep learning examples" benchmarks and parsing the output.
"""

# pylint: disable=duplicate-code

import json
import logging
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import pandas as pd

from benchmark_dockerfiles import RESNET50_DOCKERFILE
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

_RESNET50_BENCHMARK_VERSION = "0.2.0"
"""
# Version History

## 0.2.0 - (2026-02-03)
* Switched to base image w/CUDA 11.4.3 to support Kepler era cards.

## 0.1.0 - (2026-01-20)
* First version 
"""


class _ResNet50Params(NamedTuple):
    """
    Defines the knobs that can be turned for a ResNet50 Benchmark run.
    """

    batch_size: int

    mode_training: bool
    """
    Set to True to benchmark training performance, False to benchmark inference.
    """


def _parse_report_file(mode_training: bool, container_output: ContainerOutputs) -> float:
    """
    Parse a report file to the standard set of numerical results.
    :param container_output: Contains the path to the report file on disk.
    :return: Numerical results
    """

    with open(container_output.file, encoding="utf-8", mode="r") as report_file:
        loaded_dicts = [json.loads(line.replace("DLLL ", "")) for line in report_file.readlines()]
        complete_df = pd.DataFrame.from_records(loaded_dicts)

    data_only_df = pd.DataFrame.from_records(complete_df[complete_df["type"] == "LOG"]["data"])

    summary_dict = (
        data_only_df["train.total_ips" if mode_training else "val.total_ips"]
        .dropna()
        .describe()
        .to_dict()
    )

    output = summary_dict["mean"]

    return float(output)


def create_resnet50_executor(
    benchmark_name: BenchmarkName,
    gpus: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some resnet50 benchmarks.
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
    :return: The callable to run the benchmark, None if the input name is not a resnet benchmark.
    """

    name_to_parameters = {
        BenchmarkName.resnet50_train_batch_1: _ResNet50Params(mode_training=True, batch_size=1),
        BenchmarkName.resnet50_train_batch_64: _ResNet50Params(mode_training=True, batch_size=64),
        BenchmarkName.resnet50_infer_batch_1: _ResNet50Params(mode_training=False, batch_size=1),
        BenchmarkName.resnet50_infer_batch_256: _ResNet50Params(
            mode_training=False, batch_size=256
        ),
    }

    resnet_parameters: Optional[_ResNet50Params] = name_to_parameters.get(benchmark_name, None)

    if resnet_parameters is None:
        return None

    def run_resnet50_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the resnet50 benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = True

        benchmark_data = docker_wrapper.benchmark_dockerfile(
            dockerfile_path=RESNET50_DOCKERFILE,
            benchmark_name=benchmark_name.value,
            benchmark_version=_RESNET50_BENCHMARK_VERSION,
            gpus=gpus,
            create_runtime_env_vars=lambda runtime_gpus: [
                ("BATCH_SIZE", str(resnet_parameters.batch_size)),
                ("MODE_TRAINING", str(int(resnet_parameters.mode_training))),
                ("NUM_GPUS", str(len(runtime_gpus))),
            ],
            multi_gpu_native=multi_gpu_native,
            outputs_to_result=partial(_parse_report_file, resnet_parameters.mode_training),
            docker_cleanup=docker_cleanup,
            force_rebuild=force_rebuild,
            downloads_directory_path=downloads_directory_path,
        )

        return BenchmarkResult(
            name=benchmark_name,
            benchmark_version=_RESNET50_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Images Processed / Second",
            unit="i/s",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.native_multi_gpu_result,
            numerical_results=benchmark_data.report_file_numerical,
            gpu_power_usage=benchmark_data.report_file_gpu_power_usage,
        )

    return run_resnet50_docker
