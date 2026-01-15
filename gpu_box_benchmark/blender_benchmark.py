"""
Code for running the blender benchmarks and parsing the output.
Doesn't yet support multiple GPUs.
"""

import json
import logging
import re
from typing import List, Optional, Tuple, Union

import pandas as pd

from benchmark_dockerfiles import BLENDER_BENCHMARK_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import build_run_dockerfile_read_logs
from gpu_box_benchmark.locate_describe_hardware import CPUIdentity, GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

LOGGER = logging.getLogger(__name__)

_BLENDER_BENCHMARK_VERSION = "0.1.0"
_RUNS_PER_BENCHMARK = 3


def _parse_samples_per_minute(docker_logs: str) -> float:
    """
    Parse a report file to the standard set of numerical results.
    """

    json_str: Optional[str] = None

    try:

        # We split the logs at the first sign of the progress bars to isolate the JSON
        clean_section = docker_logs.split("0 / 100")[0].strip()

        # Now we find the last possible JSON array ending in this clean section
        start_idx = clean_section.find("[")
        end_idx = clean_section.rfind("]") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError(f"Could not find the start or end of the JSON array. Input: {docker_logs}")

        json_str = clean_section[start_idx:end_idx]

        data = json.loads(json_str)

        if isinstance(data, list) and len(data) > 0:
            return next(iter(data))["stats"]["samples_per_minute"]

        raise ValueError(f"JSON dict was structured incorrectly. Input: {docker_logs}")

    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:

        if json_str is not None:
            snippet = json_str[:100] if "json_str" in locals() else "Unknown"
        else:
            snippet = "Unknown"

        raise ValueError(f"Error parsing JSON: {e} | Snippet: {snippet}... | Input: {docker_logs}")


def create_blender_benchmark_executor(
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...], cpu: CPUIdentity
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some blender-based benchmarks.
    The args here fit the outer API.
    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :param cpu: CPU to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    if len(gpus) > 1 and benchmark_name != BenchmarkName.blender_monster_cpu:
        LOGGER.debug("Multi-GPU not yet supported in blender benchmark.")
        return None

    gpu_device_name = ' '.join([f'--device-name "{gpu.name}"' for gpu in gpus])

    name_to_parameters = {
        BenchmarkName.blender_monster_cpu: [
            ("RUN_ENV", f'--blender-version 4.5.0 --device-type CPU --json --verbosity 0 monster'),
        ],
        BenchmarkName.blender_monster_gpu: [
            ("RUN_ENV",
             f"--blender-version 4.5.0  --device-type CUDA {gpu_device_name} --json --verbosity 0 monster"),
        ],
    }

    blender_benchmark_envs: Optional[List[Tuple[str, Union[str, int, float, bool]]]] = (
        name_to_parameters.get(benchmark_name, None)
    )

    if blender_benchmark_envs is None:
        return None

    def run_blender_benchmark_docker() -> NumericalBenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        samples_per_minute = [
            _parse_samples_per_minute(
                build_run_dockerfile_read_logs(
                    dockerfile_path=BLENDER_BENCHMARK_DOCKERFILE,
                    tag=benchmark_name.value,
                    gpus=gpus,
                    env_vars=blender_benchmark_envs,
                )
            )
            for _ in range(_RUNS_PER_BENCHMARK)
        ]

        summary_dict = pd.Series(samples_per_minute).dropna().describe().to_dict()

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
            name=benchmark_name.value,
            benchmark_version=_BLENDER_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Samples / Minute",
            unit="s/min",
            sample_count=results.sample_count,
            mean=results.mean,
            std=results.std,
            result_min=results.result_min,
            percentile_25=results.percentile_25,
            percentile_50=results.percentile_50,
            percentile_75=results.percentile_75,
            result_max=results.result_max,
        )

    return run_blender_benchmark_docker
