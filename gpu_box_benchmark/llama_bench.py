"""
Code for running the "nvidia deep learning examples" benchmarks and parsing the output.

There's a bit of duplicated code here. I'm waiting to do so abstraction once I have the complete
suite implemented.
"""

# pylint: disable=duplicate-code

import json
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import docker
import pandas as pd
from docker.errors import APIError, BuildError

from benchmark_dockerfiles import LLAMA_BENCH_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.locate_describe_gpu import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

LOGGER = logging.getLogger(__name__)

_LLAMA_BENCH_VERSION = "0.1.0"

_NUM_TEST_TOKENS = 2048


class _LlamaBenchParams(NamedTuple):
    """
    Defines the knobs that can be turned for a ResNet50 Benchmark run.
    """

    internal_model_path: str
    prompt_tokens: int
    generation_tokens: int


def _parse_docker_logs(docker_logs: str) -> ReportFileNumerical:
    """
    Parse a report file to the standard set of numerical results.
    :param docker_logs: Logs from the docker container as a string! These logs contain our results
    and we need to extract.
    :return: Numerical results
    """

    start = docker_logs.find("[")
    end = docker_logs.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in log")

    json_blob = docker_logs[start : end + 1]
    loaded = next(iter(json.loads(json_blob)))

    summary_dict = (
        pd.Series(loaded["samples_ts"]).dropna().describe().to_dict()  # Tokens/Sample results.
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


def build_run_dockerfile_read_logs(
    dockerfile_path: Path,
    tag: str,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, Union[str, float, bool, int]]],
) -> Optional[str]:
    """
    Builds and runs a dockerfile, returning the logs from the container.
    :param dockerfile_path: Path to the dockerfile. Parent to this path will be used as the build
    directory.
    :param tag: Image will be assigned this tag.
    :param gpus: Docker container will have access to these GPUs.
    :param env_vars: List of Tuples of string and whatever that will be passed as environment
    variables to the container at runtime.
    :return: Logs from the container.
    """

    LOGGER.debug("Building Image")

    client = docker.from_env()

    build_directory = dockerfile_path.parent

    try:
        image, _build_logs = client.images.build(
            path=str(build_directory),
            dockerfile=str(dockerfile_path),
            nocache=False,
            tag=tag,
        )

    except BuildError as e:
        LOGGER.error("Docker build failed!\n")

        for entry in e.build_log:
            # entries are dicts with keys like 'stream', 'error', 'status'
            if "stream" in entry:
                LOGGER.error(entry["stream"])
            elif "error" in entry:
                LOGGER.error(entry["error"])

        raise  # re-raise if you want the failure to propagate

    except APIError:
        # Docker daemon / API-level failure
        LOGGER.error("Docker API error", exc_info=True)
        raise

    LOGGER.debug("Image Built. Running")

    container = client.containers.create(
        image=image,
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=list(str(gpu.id) for gpu in gpus),
                capabilities=[["gpu"]],
            )
        ],
        environment={key: str(value) for key, value in env_vars},
        ipc_mode="host",
        detach=True,
    )

    logs: Optional[str] = None

    try:
        container.start()
        result = container.wait()  # blocks until exit

        logs = container.logs(stdout=True, stderr=True).decode()
        status_code = result["StatusCode"]

        if status_code != 0:
            LOGGER.error("Container failed with exit code %s", status_code)
            LOGGER.error("Container logs:\n%s", logs)
            raise RuntimeError("Container execution failed")

        LOGGER.debug("Container completed successfully")
        LOGGER.debug("Container logs:\n%s", logs)

    finally:
        container.remove(force=True)

    return logs


def create_llama_bench_executor(
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some llama bench benchmarks.
    The args here fit the outer API.
    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :return: The callable to run the benchmark.
    """

    name_to_parameters = {
        BenchmarkName.llama_bench_tiny_model_prompt: _LlamaBenchParams(
            internal_model_path="/models/tiny_model.gguf",
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
        ),
        BenchmarkName.llama_bench_tiny_model_generation: _LlamaBenchParams(
            internal_model_path="/models/standard_model.gguf",
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
        ),
        BenchmarkName.llama_bench_standard_model_prompt: _LlamaBenchParams(
            internal_model_path="/models/standard_model.gguf",
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
        ),
        BenchmarkName.llama_bench_standard_model_generation: _LlamaBenchParams(
            internal_model_path="/models/standard_model.gguf",
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
        ),
    }

    llama_bench_parameters: Optional[_LlamaBenchParams] = name_to_parameters.get(
        benchmark_name, None
    )

    if llama_bench_parameters is None:
        return None

    def run_llama_bench_docker() -> NumericalBenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        logs = build_run_dockerfile_read_logs(
            dockerfile_path=LLAMA_BENCH_DOCKERFILE,
            tag=benchmark_name.value,
            gpus=gpus,
            env_vars=[
                ("MODEL_PATH", llama_bench_parameters.internal_model_path),
                ("NUM_PROMPT_TOKENS", llama_bench_parameters.prompt_tokens),
                ("NUM_GENERATION_TOKENS", llama_bench_parameters.generation_tokens),
            ],
        )

        results = _parse_docker_logs(docker_logs=logs)

        return NumericalBenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=_LLAMA_BENCH_VERSION,
            override_parameters={},
            larger_better=True,
            verbose_unit="Tokens / Second",
            unit="toks/s",
            sample_count=results.sample_count,
            mean=results.mean,
            std=results.std,
            result_min=results.result_min,
            percentile_25=results.percentile_25,
            percentile_50=results.percentile_50,
            percentile_75=results.percentile_75,
            result_max=results.result_max,
        )

    return run_llama_bench_docker
