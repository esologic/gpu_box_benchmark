"""
Code for running the "nvidia deep learning examples" benchmarks and parsing the output.
"""

# pylint: disable=duplicate-code

import json
import logging
from functools import partial
from typing import NamedTuple, Optional, Tuple

import pandas as pd

from benchmark_dockerfiles import RESNET50_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult, NumericalResultKey

LOGGER = logging.getLogger(__name__)

_RESNET50_BENCHMARK_VERSION = "0.1.0"


class _ResNet50Params(NamedTuple):
    """
    Defines the knobs that can be turned for a ResNet50 Benchmark run.
    """

    batch_size: int
    amp_enabled: bool

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
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...], docker_cleanup: bool
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some resnet50 benchmarks.
    The args here fit the outer API.
    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :return: The callable to run the benchmark, None if the input name is not a resnet benchmark.
    """

    name_to_parameters = {
        BenchmarkName.resnet50_train_batch_1_amp: _ResNet50Params(
            mode_training=True, batch_size=1, amp_enabled=True
        ),
        BenchmarkName.resnet50_train_batch_64_amp: _ResNet50Params(
            mode_training=True, batch_size=64, amp_enabled=True
        ),
        BenchmarkName.resnet50_infer_batch_1_amp: _ResNet50Params(
            mode_training=False, batch_size=1, amp_enabled=True
        ),
        BenchmarkName.resnet50_infer_batch_256_amp: _ResNet50Params(
            mode_training=False, batch_size=256, amp_enabled=True
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

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_RESNET50_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Images Processed / Second",
            unit="i/s",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.native_multi_gpu_result,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=RESNET50_DOCKERFILE,
                tag_prefix=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: [
                    ("BATCH_SIZE", str(resnet_parameters.batch_size)),
                    ("MODE_TRAINING", str(int(resnet_parameters.mode_training))),
                    ("NUM_GPUS", str(len(runtime_gpus))),
                ],
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=partial(_parse_report_file, resnet_parameters.mode_training),
                docker_cleanup=docker_cleanup,
            ),
        )

    return run_resnet50_docker
