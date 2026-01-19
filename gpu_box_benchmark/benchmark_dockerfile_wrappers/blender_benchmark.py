"""
Code for running the blender benchmarks and parsing the output.
Doesn't yet support multiple GPUs.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

from benchmark_dockerfiles import BLENDER_BENCHMARK_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult

LOGGER = logging.getLogger(__name__)

_BLENDER_BENCHMARK_VERSION = "0.1.0"


def _parse_samples_per_minute(container_outputs: ContainerOutputs) -> float:
    """
    Finds and parses the JSON result array from docker logs,
    even if interleaved with progress bars.
    :param container_outputs: Contains all the different outputs from the container.
    :return: Samples per minute as a float.
    """

    docker_logs = container_outputs.logs

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

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    if benchmark_name not in (BenchmarkName.blender_monster_cpu, BenchmarkName.blender_monster_gpu):
        return None

    def create_runtime_env_vars(
        runtime_gpus: Tuple[GPUIdentity, ...],
    ) -> List[Tuple[str, str]]:
        """
        Blender benchmark needs to know the name of the GPU it should run on.
        :param runtime_gpus: The GPUs the test will be run on.
        :return: Rendered environment variables.
        """

        if benchmark_name == BenchmarkName.blender_monster_cpu:
            hardware_description: str = "--device-type CPU"
        elif benchmark_name == BenchmarkName.blender_monster_gpu:

            if len(runtime_gpus) > 1:
                raise ValueError(
                    "Multiple GPUs not supported for a single Blender "
                    f"Benchmark Run. Input: {runtime_gpus}"
                )

            gpu_device_name: str = " ".join([f'--device-name "{gpu.name}"' for gpu in runtime_gpus])
            hardware_description = " ".join(["--device-type CUDA", gpu_device_name])
        else:
            raise ValueError(f"Bad benchmark type for blender benchmark: {benchmark_name}")

        output: List[Tuple[str, str]] = [
            (
                "RUN_ENV",
                " ".join(
                    [
                        "--blender-version 4.5.0",
                        "--verbosity 0",
                        "--json",
                        hardware_description,
                        "monster",
                    ]
                ),
            )
        ]

        return output

    def run_blender_benchmark_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = False

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_BLENDER_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Samples / Minute",
            unit="s/min",
            multi_gpu_native=multi_gpu_native,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=BLENDER_BENCHMARK_DOCKERFILE,
                tag=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=create_runtime_env_vars,
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_samples_per_minute,
            ),
        )

    return run_blender_benchmark_docker
