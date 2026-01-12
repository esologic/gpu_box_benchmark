"""
Code for running the "nvidia deep learning examples" benchmarks and parsing the output.
"""

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple, Optional, Tuple

import docker
import pandas as pd

from benchmark_dockerfiles import RESNET50_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.gpu_discovery import GPUDescription
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult

LOGGER = logging.getLogger(__name__)


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


class _ReportFileNumerical(NamedTuple):
    """
    Intermediate type to contain the numerical result.
    """

    sample_count: float
    mean: float
    std: float
    result_min: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    result_max: float


def _parse_report_file(report_path: Path, mode_training: bool = True) -> _ReportFileNumerical:
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

    output = _ReportFileNumerical(
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


def deep_learning_examples_lookup(
    benchmark_name: BenchmarkName, gpus: Tuple[GPUDescription, ...]
) -> Optional[BenchmarkExecutor]:
    """

    :param benchmark_name:
    :return:
    """

    print(benchmark_name, gpus)

    def output() -> NumericalBenchmarkResult:

        client = docker.from_env()

        LOGGER.info("Building Image")

        build_directory = RESNET50_DOCKERFILE.parent

        image, _logs = client.images.build(
            path=str(build_directory),
            dockerfile=str(RESNET50_DOCKERFILE),
            nocache=False,
            tag="resnet50",
        )

        LOGGER.info("Image Built. Running")

        mode_training = False

        with NamedTemporaryFile(suffix=".txt") as f:

            container = client.containers.create(
                image=image,
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=list(map(str, gpus)),
                        capabilities=[["gpu"]],
                    )
                ],
                environment={
                    "BATCH_SIZE": "256",
                    "MODE_TRAINING": str(int(mode_training)),
                    "NUM_GPUS": str(len(gpus)),
                },
                volumes={
                    str(f.name): {
                        "bind": "/results/output.txt",
                        "mode": "rw",
                    }
                },
                detach=True,
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

                LOGGER.info("Container completed successfully")
                LOGGER.debug("Container logs:\n%s", logs)

            finally:
                container.remove(force=True)

            f.seek(0)

            results = _parse_report_file(Path(f.name), mode_training=mode_training)

        return NumericalBenchmarkResult(
            name=benchmark_name.value,
            benchmark_version="1.0.0",
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

    return output
