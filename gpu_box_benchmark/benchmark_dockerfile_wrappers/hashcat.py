"""
Code for running hashcat in benchmark mode and parsing the output.
"""

import logging
from typing import Dict, List, Optional, Tuple

from benchmark_dockerfiles import HASHCAT_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult, NumericalResultKey

LOGGER = logging.getLogger(__name__)

_HASHCAT_BENCHMARK_VERSION = "0.1.0"


def _parse_megahashes_per_second(container_outputs: ContainerOutputs) -> float:
    """
    Parses machine-readable hashcat logs for multiple GPUs.
    Sums the H/s (final field) from all valid data rows and converts to MH/s.
    :param container_outputs: Contains logs.
    :return: Total summed MH/s.
    """

    total_hs = sum(
        (
            float(line.split(":")[-1])
            for line in container_outputs.logs.strip().splitlines()
            if (":" in line) and len(line.split(":")) > 5
        )
    )

    if not total_hs:
        raise ValueError(f"No valid Hashcat benchmark rows found. Logs: {container_outputs.logs}")

    # Convert raw H/s to MH/s
    return total_hs / 1_000_000


def create_hashcat_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...], docker_cleanup: bool
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run hashcat in benchmark mode.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :return: The callable to run the benchmark.
    """

    name_to_env_vars: Dict[BenchmarkName, List[Tuple[str, str]]] = {
        BenchmarkName.hashcat_sha256: [
            ("HASH_TYPE", "1400"),
        ],
    }

    run_env_vars: Optional[List[Tuple[str, str]]] = name_to_env_vars.get(benchmark_name, None)

    if run_env_vars is None:
        return None

    def run_hashcat_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = True

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_HASHCAT_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Mega Hashes / Second",
            unit="MH/s",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.forced_multi_gpu_sum,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=HASHCAT_DOCKERFILE,
                tag_prefix=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: run_env_vars,
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_megahashes_per_second,
                docker_cleanup=docker_cleanup,
            ),
        )

    return run_hashcat_docker
