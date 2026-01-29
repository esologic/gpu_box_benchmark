"""
Code for running `gdsio` and parsing the output.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from benchmark_dockerfiles import NVIDIA_GDS_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult, NumericalResultKey

LOGGER = logging.getLogger(__name__)

_GDSIO_BENCHMARK_VERSION = "0.1.0"


def _parse_throughput(container_outputs: ContainerOutputs) -> float:
    """
    Finds and returns the run throughput from gdsio logs.
    :param container_outputs: All outputs from the container, contains the logs which are the full
    output from the docker container post run.
    :return: Final score as a float (GiB/sec).
    """
    lines = container_outputs.logs.splitlines()

    # Filter for the line containing the 'Throughput:' metric
    throughput_line = next((line for line in lines if "Throughput:" in line), None)

    if not throughput_line:
        raise ValueError("Could not find 'Throughput:' in container logs.")

    # Use regex to find the float value following 'Throughput: '
    # \d+\.\d+ matches digits, a dot, and more digits
    match = re.search(r"Throughput:\s+(\d+\.\d+)", throughput_line)

    if match:
        return float(match.group(1))

    raise ValueError(f"Found throughput line but could not parse value: {throughput_line}")


def create_gdsio_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...], docker_cleanup: bool
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some gdsio performance benchmarks.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :return: The callable to run the benchmark.
    """

    name_to_env_vars: Dict[BenchmarkName, List[Tuple[str, str]]] = {
        BenchmarkName.gdsio_type_0: [
            ("IO_TYPE", "0"),
        ],
        BenchmarkName.gdsio_type_2: [
            ("IO_TYPE", "2"),
        ],
    }

    run_env_vars: Optional[List[Tuple[str, str]]] = name_to_env_vars.get(benchmark_name, None)

    if run_env_vars is None:
        return None

    def run_gdsio_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = False

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_GDSIO_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Throughput GiB / Second",
            unit="GiB/sec",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.forced_multi_gpu_sum,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=NVIDIA_GDS_DOCKERFILE,
                tag_prefix=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: run_env_vars,
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_throughput,
                docker_cleanup=docker_cleanup,
            ),
        )

    return run_gdsio_docker
