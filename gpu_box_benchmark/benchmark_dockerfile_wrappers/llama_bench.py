"""
Code for running the llama-bench benchmarks in llama.cpp and parsing the output.
"""

import json
import logging
from typing import NamedTuple, Optional, Tuple

import pandas as pd

from benchmark_dockerfiles import LLAMA_BENCH_DOCKERFILE
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import build_run_dockerfile_read_logs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

LOGGER = logging.getLogger(__name__)

_LLAMA_BENCH_VERSION = "0.1.0"

_NUM_TEST_TOKENS = 512


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
        percentile_25=summary_dict["25%"],
        percentile_50=summary_dict["50%"],
        sample_count=summary_dict["count"],
        mean=summary_dict["mean"],
        std=summary_dict["std"],
        result_min=summary_dict["min"],
        percentile_75=summary_dict["75%"],
        result_max=summary_dict["max"],
    )

    return output


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
