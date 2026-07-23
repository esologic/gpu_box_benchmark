"""
Code for running hashcat in benchmark mode and parsing the output.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark_dockerfiles import HASHCAT_DOCKERFILE
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

_HASHCAT_BENCHMARK_VERSION = "0.1.0"
"""
# Version History

## 0.2.0 - (2026-02-03)
* Initial introduction.
"""


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
    benchmark_name: BenchmarkName,
    gpus: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run hashcat in benchmark mode.
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
            name=benchmark_name,
            benchmark_version=_HASHCAT_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Mega Hashes / Second",
            unit="MH/s",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.forced_multi_gpu_sum,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=HASHCAT_DOCKERFILE,
                benchmark_name=benchmark_name.value,
                benchmark_version=_HASHCAT_BENCHMARK_VERSION,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: run_env_vars,
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_megahashes_per_second,
                docker_cleanup=docker_cleanup,
                force_rebuild=force_rebuild,
                downloads_directory_path=downloads_directory_path,
            ),
        )

    return run_hashcat_docker
