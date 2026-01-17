"""
Code for running the "nvidia deep learning examples" benchmarks and parsing the output.
"""

# pylint: disable=duplicate-code

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional, Tuple

import docker
import pandas as pd

from benchmark_dockerfiles import RESNET50_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

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


def _parse_report_file(report_path: Path, mode_training: bool) -> ReportFileNumerical:
    """
    Parse a report file to the standard set of numerical results.
    :param report_path: Path to the report file on disk.
    :return: Numerical results
    """

    with open(report_path, encoding="utf-8", mode="r") as report_file:
        loaded_dicts = [json.loads(line.replace("DLLL ", "")) for line in report_file.readlines()]
        complete_df = pd.DataFrame.from_records(loaded_dicts)

    data_only_df = pd.DataFrame.from_records(complete_df[complete_df["type"] == "LOG"]["data"])

    summary_dict = (
        data_only_df["train.total_ips" if mode_training else "val.total_ips"]
        .dropna()
        .describe()
        .to_dict()
    )

    output = ReportFileNumerical(
        sample_count=summary_dict["count"],
        mean=summary_dict["mean"],
        std=summary_dict["std"],
        result_min=summary_dict["min"],
        percentile_25=summary_dict["25%"],
        percentile_50=summary_dict["50%"],
        percentile_75=summary_dict["75%"],
        result_max=summary_dict["max"],
    )

    return output


def create_resnet50_executor(
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some resnet50 benchmarks.
    The args here fit the outer API.
    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
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

    def run_resnet50_docker() -> NumericalBenchmarkResult:
        """
        Build and run the docker image that runs the resnet50 benchmark.
        :return: Parsed results.
        """

        client = docker.from_env()

        LOGGER.debug("Building Image")

        build_directory = RESNET50_DOCKERFILE.parent

        image, _logs = client.images.build(
            path=str(build_directory),
            dockerfile=str(RESNET50_DOCKERFILE),
            nocache=False,
            tag=benchmark_name.value,
        )

        LOGGER.debug("Image Built. Running")

        with NamedTemporaryFile(suffix=".txt") as temporary_file:

            container = client.containers.create(
                image=image,
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=list(str(gpu.id) for gpu in gpus),
                        capabilities=[["gpu"]],
                    )
                ],
                environment={
                    key: str(value)
                    for key, value in [
                        ("BATCH_SIZE", resnet_parameters.batch_size),
                        ("MODE_TRAINING", int(resnet_parameters.mode_training)),
                        ("NUM_GPUS", len(gpus)),
                    ]
                },
                volumes={
                    str(temporary_file.name): {
                        # This location is baked into the dockerfile.
                        "bind": "/results/output.txt",
                        "mode": "rw",
                    }
                },
                detach=True,
                ipc_mode="host",
            )

            try:
                container.start()
                result = container.wait()  # blocks until exit

                exit_code = result["StatusCode"]
                logs = container.logs(stdout=True, stderr=True).decode()

                if exit_code != 0:
                    LOGGER.error("Container failed with exit code %s", exit_code)
                    LOGGER.error("Container logs:\n%s", logs)
                    raise RuntimeError("Container execution failed")

                LOGGER.debug("Container completed successfully")
                LOGGER.debug("Container logs:\n%s", logs)

            finally:
                container.remove(force=True)

            temporary_file.seek(0)

            results = _parse_report_file(
                Path(temporary_file.name), mode_training=resnet_parameters.mode_training
            )

        return NumericalBenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_RESNET50_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Images Processed / Second",
            unit="i/s",
            sample_count=results.sample_count,
            mean=results.mean,
            std=results.std,
            result_min=results.result_min,
            percentile_25=results.percentile_25,
            percentile_50=results.percentile_50,
            percentile_75=results.percentile_75,
            result_max=results.result_max,
        )

    return run_resnet50_docker
