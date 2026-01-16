"""
Code for running the blender benchmarks and parsing the output.
Doesn't yet support multiple GPUs.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from benchmark_dockerfiles import BLENDER_BENCHMARK_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import build_run_dockerfile_read_logs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

LOGGER = logging.getLogger(__name__)

_BLENDER_BENCHMARK_VERSION = "0.1.0"
_RUNS_PER_BENCHMARK = 3


def _parse_samples_per_minute(docker_logs: str) -> float:
    """
    Finds and parses the JSON result array from docker logs,
    even if interleaved with progress bars.
    """
    try:
        # Regex to find a JSON array: starts with [ and ends with ]
        # We use re.DOTALL so that '.' matches newlines
        # We use non-greedy matching '.*?' to find the most likely JSON block
        json_match = re.search(r"(\[\s*\{.*\}\s*\])", docker_logs, re.DOTALL)

        if not json_match:
            raise ValueError(f"Could not locate a JSON array in the logs. Input {docker_logs}")

        json_str = json_match.group(1)

        # Sometimes progress bar fragments get caught in the greedy match
        # if the logs are very messy. We refine the string by finding the
        # actual first '[' and last ']'
        start_idx = json_str.find("[")
        end_idx = json_str.rfind("]") + 1
        json_str = json_str[start_idx:end_idx]

        data = json.loads(json_str)

        if isinstance(data, list) and len(data) > 0:
            return float(data[0]["stats"]["samples_per_minute"])

        raise ValueError(f"JSON was not a list or was empty. Input {docker_logs}")

    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"Error parsing JSON: {e} Input {docker_logs}") from e


def create_blender_benchmark_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some blender-based benchmarks.
    The args here fit the outer API.

    TODO: We should probably just run once per GPU.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    if len(gpus) > 1 and benchmark_name != BenchmarkName.blender_monster_cpu:
        LOGGER.debug("Multi-GPU not yet supported in blender benchmark.")
        return None

    gpu_device_name = " ".join([f'--device-name "{gpu.name}"' for gpu in gpus])

    name_to_parameters: Dict[BenchmarkName, List[Tuple[str, Union[str, float, bool, int]]]] = {
        BenchmarkName.blender_monster_cpu: [
            ("RUN_ENV", "--blender-version 4.5.0 --device-type CPU --json --verbosity 0 monster"),
        ],
        BenchmarkName.blender_monster_gpu: [
            (
                "RUN_ENV",
                (
                    "--blender-version 4.5.0  --device-type "
                    f"CUDA {gpu_device_name} --json --verbosity 0 monster"
                ),
            ),
        ],
    }

    blender_benchmark_envs: Optional[List[Tuple[str, Union[str, float, bool, int]]]] = (
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
            percentile_50=results.percentile_50,
            percentile_75=results.percentile_75,
            result_max=results.result_max,
            name=benchmark_name.value,
            benchmark_version=_BLENDER_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Samples / Minute",
            unit="s/min",
            sample_count=results.sample_count,
            std=results.std,
            mean=results.mean,
            result_min=results.result_min,
            percentile_25=results.percentile_25,
        )

    return run_blender_benchmark_docker
