"""
Code for running the FAHBench and parsing the output.
Doesn't yet support multiple GPUs.
"""

import logging
from typing import Dict, List, Optional, Tuple

from benchmark_dockerfiles import FAHBENCH_BENCHMARK_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult, NumericalResultKey

LOGGER = logging.getLogger(__name__)

_FAH_BENCHMARK_VERSION = "0.1.0"


def _parse_final_score(container_outputs: ContainerOutputs) -> float:
    """
    Finds and returns the final score from some FAHBench logs.
    :param container_outputs: All outputs from the container, contains the logs which are the full
    output from the docker container post run.
    :return: Final score.
    """

    lines = container_outputs.logs.splitlines()
    final_score_line = next(iter([line for line in lines if "Final score" in line]))
    final_score = float(final_score_line.split(":")[1].strip())
    return final_score


def create_fah_bench_executor(  # pylin
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some Folding@Home (FAHBench) benchmarks.
    The args here fit the outer API.

    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    name_to_parameters: Dict[BenchmarkName, List[Tuple[str, str]]] = {
        BenchmarkName.fah_bench_single: [
            ("FAHBENCH_PRECISION", "single"),
        ],
        BenchmarkName.fah_bench_double: [
            ("FAHBENCH_PRECISION", "double"),
        ],
    }

    envs: Optional[List[Tuple[str, str]]] = name_to_parameters.get(benchmark_name, None)

    if envs is None:
        return None

    def run_fah_bench_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = False

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_FAH_BENCHMARK_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Nanoseconds / Day",
            unit="ns/day",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.forced_multi_gpu_sum,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=FAHBENCH_BENCHMARK_DOCKERFILE,
                tag_prefix=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: envs,
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_final_score,
            ),
        )

    return run_fah_bench_docker
