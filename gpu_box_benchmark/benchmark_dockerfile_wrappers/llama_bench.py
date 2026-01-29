"""
Code for running the llama-bench benchmarks in llama.cpp and parsing the output.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import pandas as pd

from benchmark_dockerfiles import IK_LLAMA_BENCH_DOCKERFILE, LLAMA_BENCH_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_jobs import BenchmarkExecutor, BenchmarkName
from gpu_box_benchmark.docker_wrapper import ContainerOutputs
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult, NumericalResultKey

LOGGER = logging.getLogger(__name__)

_LLAMA_BENCH_VERSION = "0.1.0"
_IK_LLAMA_BENCH_VERSION = "0.1.0"

_NUM_TEST_TOKENS = 512


class _LlamaBenchParams(NamedTuple):
    """
    Defines the knobs that can be turned for a ResNet50 Benchmark run.
    """

    internal_model_path: str
    prompt_tokens: int
    generation_tokens: int


def _parse_docker_logs(container_outputs: ContainerOutputs) -> float:
    """
    Parse a report file to the standard set of numerical results.
    Handles standard docker logs mixed into the JSON output stream.
    :param container_outputs: Contains the logs from the docker container as a string! These logs
    contain our results, and we need to extract.
    :return: Numerical results
    """
    docker_logs = container_outputs.logs

    start = docker_logs.find("[")
    end = docker_logs.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in log. Complete Logs: {docker_logs}")

    results = []
    decoder = json.JSONDecoder()
    current_pos = start

    while current_pos <= end:
        # Find the start of the next JSON object
        next_brace = docker_logs.find("{", current_pos, end + 1)

        # If no more braces are found, stop
        if next_brace == -1:
            break

        try:
            # raw_decode extracts one valid object and returns the index where it stopped
            # It ignores trailing garbage (like "~ggml_backend...").
            obj, index_end = decoder.raw_decode(docker_logs, idx=next_brace)
            results.append(obj)

            # Move our search position to the end of the object we just found
            current_pos = index_end
        except json.JSONDecodeError:
            # If the brace we found wasn't valid JSON, skip it and continue searching
            current_pos = next_brace + 1

    if not results:
        raise ValueError(
            f"Found bounds [...] but no valid JSON objects inside. Logs: {docker_logs}"
        )

    loaded = results[0]

    summary_dict = (
        pd.Series(loaded["samples_ts"]).dropna().describe().to_dict()  # Tokens/Sample results.
    )

    return float(summary_dict["mean"])


def create_llama_bench_executor(
    benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...], docker_cleanup: bool
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some llama bench benchmarks.
    The args here fit the outer API.
    :param benchmark_name: To lookup.
    :param gpus: GPUs to use in the benchmark.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :return: The callable to run the benchmark.
    """

    version: Optional[str] = None
    dockerfile_path: Optional[Path] = None

    if benchmark_name in (
        BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt,
        BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation,
        BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt,
        BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation,
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt,
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation,
        BenchmarkName.llama_bench_open_mistral_moe_prompt,
        BenchmarkName.llama_bench_open_mistral_moe_generation,
    ):
        version = _LLAMA_BENCH_VERSION
        dockerfile_path = LLAMA_BENCH_DOCKERFILE
    elif benchmark_name in (
        BenchmarkName.ik_llama_bench_meta_llama_3_8b_instruct_prompt,
        BenchmarkName.ik_llama_bench_meta_llama_3_8b_instruct_generation,
        BenchmarkName.ik_llama_bench_qwen_1_5_moe_chat_prompt,
        BenchmarkName.ik_llama_bench_qwen_1_5_moe_chat_generation,
    ):
        version = _IK_LLAMA_BENCH_VERSION
        dockerfile_path = IK_LLAMA_BENCH_DOCKERFILE
    else:
        return None

    name_to_parameters = {
        # --- Qwen 2.5 1.5B (Dense) ---
        BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt: _LlamaBenchParams(
            internal_model_path="/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
        ),
        BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation: _LlamaBenchParams(
            internal_model_path="/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
        ),
        # --- Meta Llama 3 8B (Dense) ---
        BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt: _LlamaBenchParams(
            internal_model_path="/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
        ),
        BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation: _LlamaBenchParams(
            internal_model_path="/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
        ),
        # --- Qwen 1.5 MoE A2.7B (Sparse MoE) ---
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt: _LlamaBenchParams(
            internal_model_path="/models/Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf",
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
        ),
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation: _LlamaBenchParams(
            internal_model_path="/models/Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf",
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
        ),
        # --- OpenMistral MoE (Sparse MoE) ---
        BenchmarkName.llama_bench_open_mistral_moe_prompt: _LlamaBenchParams(
            internal_model_path="/models/OpenMistral-MoE-Q2_K.gguf",
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
        ),
        BenchmarkName.llama_bench_open_mistral_moe_generation: _LlamaBenchParams(
            internal_model_path="/models/OpenMistral-MoE-Q2_K.gguf",
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
        ),
    }

    # These have the exact same parameters but will use a different dockerfile.
    # The mistral models don't work with ik_llama.
    name_to_parameters[BenchmarkName.ik_llama_bench_meta_llama_3_8b_instruct_prompt] = (
        name_to_parameters[BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt]
    )
    name_to_parameters[BenchmarkName.ik_llama_bench_meta_llama_3_8b_instruct_generation] = (
        name_to_parameters[BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation]
    )
    name_to_parameters[BenchmarkName.ik_llama_bench_qwen_1_5_moe_chat_prompt] = name_to_parameters[
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt
    ]
    name_to_parameters[BenchmarkName.ik_llama_bench_qwen_1_5_moe_chat_generation] = (
        name_to_parameters[BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation]
    )

    llama_bench_parameters: Optional[_LlamaBenchParams] = name_to_parameters.get(
        benchmark_name, None
    )

    if llama_bench_parameters is None:
        return None

    def run_llama_bench_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = True

        return BenchmarkResult(
            name=benchmark_name.value,
            benchmark_version=version,
            override_parameters={},
            larger_better=True,
            verbose_unit="Tokens / Second",
            unit="toks/s",
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.native_multi_gpu_result,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=dockerfile_path,
                tag_prefix=benchmark_name.value,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: [
                    ("MODEL_PATH", str(llama_bench_parameters.internal_model_path)),
                    ("NUM_PROMPT_TOKENS", str(llama_bench_parameters.prompt_tokens)),
                    ("NUM_GENERATION_TOKENS", str(llama_bench_parameters.generation_tokens)),
                ],
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=_parse_docker_logs,
                docker_cleanup=docker_cleanup,
            ),
        )

    return run_llama_bench_docker
