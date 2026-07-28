"""
Code for running the llama-bench benchmarks in llama.cpp and parsing the output.

There is also a WIP dockerfile for ik_llama as well. This project was abandoned because it's
CUDA20 requirement is incompatible with many of the old Tesla Cards. As more important test are
fundamentally incompatible with older hardware, we'll have to be able to have different suite modes
but until then ik_llama isn't supported.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from benchmark_dockerfiles import LLAMA_BENCH_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_types import (
    BenchmarkExecutor,
    BenchmarkName,
    BenchmarkResult,
    NumericalResultKey,
)
from gpu_box_benchmark.docker_wrapper import ContainerOutputs, OutputsToResult
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity

LOGGER = logging.getLogger(__name__)

_LLAMA_BENCH_PERFORMANCE_VERSION = "0.4.0"
"""
The benchmarks related to Tok/S throughput.

# Version History

## 0.4.0 - (2026-07-22)
* Updated runner script to optionally run multiple models to enable the max parameters benchmark.
* Touched a bunch of code to decrease redundancy between the two testing modalities.

## 0.3.0 - (2026-05-07)
* Added CUDA architecture 75 to support Tesla T40.

## 0.2.0 - (2026-02-03)
* Switched to base image w/CUDA 11.4.3 to support Kepler era cards.
* Set of models is now: Qwen2.5 1.5B, Llama3 8B, Qwen1.5 MoE 7B. 

## 0.1.0 - (2026-01-20)
* First version 
"""

_LLAMA_BENCH_MAX_MODEL_PARAMETERS_VERSION = "0.1.0"
"""
The benchmarks related to seeing the largest model that can fit in the GPU.

# Version History

## 0.1.0 - (2026-07-21)
* First version 
"""

_NUM_TEST_TOKENS = 512
_NUM_REPETITIONS = 10


class _HFModelSpec(BaseModel):
    """
    Describes a model available on huggingface to be used for benchmarking.
    """

    repo: str
    """
    HF repo.
    """

    download: str
    """
    Item to download in the repo, can contain wildcard to get multipart models.
    """

    entrypoint: str
    """
    Name of model to run. For multipart models, the first part must be used.
    """


_QWEN_2_5_1_5B_MODEL = _HFModelSpec(
    repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    download="qwen2.5-1.5b-instruct-q4_k_m*.gguf",
    entrypoint="qwen2.5-1.5b-instruct-q4_k_m.gguf",
)

_META_LLAMA_3_8B_MODEL = _HFModelSpec(
    repo="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    download="Meta-Llama-3-8B-Instruct-Q4_K_M*.gguf",
    entrypoint="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

_QWEN_1_5_MOE_MODEL = _HFModelSpec(
    repo="tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF",
    download="Qwen1.5-MoE-A2.7B-Chat-Q2_K*.gguf",
    entrypoint="Qwen1.5-MoE-A2.7B-Chat-Q2_K.gguf",
)

# Performance
_QWEN_2_5_72B_Q4 = _HFModelSpec(
    repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
    download="qwen2.5-72b-instruct-q4_k_m*.gguf",
    entrypoint="qwen2.5-72b-instruct-q4_k_m-00001-of-00012.gguf",
)

# Performance
_QWEN_2_5_32B_Q4 = _HFModelSpec(
    repo="Qwen/Qwen2.5-32B-Instruct-GGUF",
    download="qwen2.5-32b-instruct-q4_k_m*.gguf",
    entrypoint="qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf",
)

_Q4_SPECS = [
    _HFModelSpec(
        repo="bartowski/Gigaberg-Mistral-Large-123B-GGUF",
        download="Gigaberg-Mistral-Large-123B-Q4_K_M*.gguf",
        entrypoint=(
            "Gigaberg-Mistral-Large-123B-Q4_K_M/"
            "Gigaberg-Mistral-Large-123B-Q4_K_M-00001-of-00002.gguf"
        ),
    ),
    _QWEN_2_5_72B_Q4,
    _QWEN_2_5_32B_Q4,
    _HFModelSpec(
        repo="bartowski/gemma-2-27b-it-GGUF",
        download="gemma-2-27b-it-Q4_K_M*.gguf",
        entrypoint="gemma-2-27b-it-Q4_K_M.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-14B-Instruct-GGUF",
        download="qwen2.5-14b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-14b-instruct-q4_k_m.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        download="qwen2.5-7b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        download="qwen2.5-3b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        download="qwen2.5-1.5b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        download="qwen2.5-0.5b-instruct-q4_k_m*.gguf",
        entrypoint="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ),
]

# Should be 16B
_DEEPSEEK_CODER_V2_LITE = _HFModelSpec(
    repo="lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
    download="DeepSeek-Coder-V2-Lite-Instruct-Q8*.gguf",
    entrypoint="DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf",
)

_GEMMA_4_Q8 = _HFModelSpec(
    repo="unsloth/gemma-4-31B-it-GGUF",
    download="gemma-4-31B-it-Q8*.gguf",
    entrypoint="gemma-4-31B-it-Q8_0.gguf",
)

_Q8_SPECS = [
    _HFModelSpec(
        repo="bartowski/Gigaberg-Mistral-Large-123B-GGUF",
        download="Gigaberg-Mistral-Large-123B-Q8_0*.gguf",
        entrypoint=(
            "Gigaberg-Mistral-Large-123B-Q8_0/"
            "Gigaberg-Mistral-Large-123B-Q8_0-00001-of-00004.gguf"
        ),
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-72B-Instruct-GGUF",
        download="qwen2.5-72b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-72b-instruct-q8_0-00001-of-00021.gguf",
    ),
    _GEMMA_4_Q8,
    _HFModelSpec(
        repo="Qwen/Qwen2.5-32B-Instruct-GGUF",
        download="qwen2.5-32b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf",
    ),
    _HFModelSpec(
        repo="bartowski/gemma-2-27b-it-GGUF",
        download="gemma-2-27b-it-Q8_0*.gguf",
        entrypoint="gemma-2-27b-it-Q8_0.gguf",
    ),
    # 16B Parameter Model
    _DEEPSEEK_CODER_V2_LITE,
    _HFModelSpec(
        repo="Qwen/Qwen2.5-14B-Instruct-GGUF",
        download="qwen2.5-14b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-14b-instruct-q8_0.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        download="qwen2.5-7b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
        download="qwen2.5-3b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-3b-instruct-q8_0.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        download="qwen2.5-1.5b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-1.5b-instruct-q8_0.gguf",
    ),
    _HFModelSpec(
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        download="qwen2.5-0.5b-instruct-q8_0*.gguf",
        entrypoint="qwen2.5-0.5b-instruct-q8_0.gguf",
    ),
]

_QWEN_3_6_35B_F16 = _HFModelSpec(
    repo="unsloth/Qwen3.6-35B-A3B-GGUF",
    download="BF16/Qwen3.6-35B-A3B-BF16-*.gguf",
    entrypoint="BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
)

_GEMMA_4_16 = _HFModelSpec(
    repo="unsloth/gemma-4-31B-it-GGUF",
    download="BF16/gemma-4-31B-it-BF16-*.gguf",
    entrypoint="BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf",
)

_QWEN_3_6_27B_F16 = _HFModelSpec(
    repo="unsloth/Qwen3.6-27B-GGUF",
    download="BF16/Qwen3.6-27B-BF16-*.gguf",
    entrypoint="BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf",
)

_QWEN_2_5_14B_F16 = _HFModelSpec(
    repo="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
    download="qwen2.5-coder-14b-instruct-fp16-*.gguf",
    entrypoint="qwen2.5-coder-14b-instruct-fp16-00001-of-00004.gguf",
)

_F16_SPECS = [
    _QWEN_3_6_35B_F16,
    _GEMMA_4_16,
    _QWEN_3_6_27B_F16,
    _QWEN_2_5_14B_F16,
    _HFModelSpec(
        repo="unsloth/gemma-4-12b-it-GGUF",
        download="gemma-4-12b-it-BF16*.gguf",
        entrypoint="gemma-4-12b-it-BF16.gguf",
    ),
    # Should be 8B parameters.
    _HFModelSpec(
        repo="unsloth/gemma-4-E4B-it-GGUF",
        download="gemma-4-E4B-it-BF16*.gguf",
        entrypoint="gemma-4-E4B-it-BF16.gguf",
    ),
]

_CONTEXT_DEPTH_4K = 4096
_CONTEXT_DEPTH_32K = 32768


class _LlamaBenchParams(NamedTuple):
    """
    Defines the knobs that can be turned for a ResNet50 Benchmark run.
    """

    model_specs: List[_HFModelSpec]
    prompt_tokens: int
    generation_tokens: int
    context_depth: int
    repetitions: int


class _LlamaBenchOutputs(NamedTuple):
    """
    Different outputs from llama.cpp that we care about from the runs.
    """

    tokens_per_second: float
    model_number_parameters: int


def _parse_docker_logs(container_outputs: ContainerOutputs) -> _LlamaBenchOutputs:
    """
    Parse a report file to the standard set of numerical results.
    Handles standard docker logs mixed into the JSON output stream.
    :param container_outputs: Contains the logs from the docker container as a string! These logs
    contain our results, and we need to extract.
    :return: Numerical results as an NT that will be parsed later.
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
        pd.Series(loaded["samples_ts"]).dropna().describe().to_dict()  # Tokens/Second results.
    )

    return _LlamaBenchOutputs(
        tokens_per_second=float(summary_dict["mean"]),
        model_number_parameters=loaded["model_n_params"],
    )


def _abstract_llama_bench_executor(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    benchmark_name: BenchmarkName,
    version: str,
    name_to_parameters: Dict[BenchmarkName, _LlamaBenchParams],
    verbose_unit: str,
    unit: str,
    outputs_to_results: OutputsToResult,
    gpus: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> Optional[BenchmarkExecutor]:
    """
    Creates a llama.cpp benchmark executor backed by the generic Docker benchmark
    infrastructure.

    This helper encapsulates the common logic for the various llama.cpp benchmark
    families. Callers provide the set of supported benchmark names, benchmark-specific
    parameters, Docker image configuration, and a parser that extracts the desired
    numerical result from the container output.

    :param benchmark_name: Benchmark to create an executor for.
    :param version: Benchmark implementation version.
    :param name_to_parameters: Mapping from benchmark name to the llama.cpp model and runtime
    parameters used for that benchmark.
    :param verbose_unit: Human-readable unit for the reported result.
    :param unit: Short unit string for the reported result.
    :param outputs_to_results: Parses the benchmark container output into the numerical result
    reported by the benchmark.
    :param gpus: GPUs to use when running the benchmark.
    :param docker_cleanup: If true, remove the benchmark Docker image after benchmarking.
    :param force_rebuild: If given, the underlying docker containers are rebuilt from scratch
    before running. This will take more time that using the existing images.
    :param downloads_directory_path: Benchmark Docker containers may need to download test assets
    during build time or runtime. If not given, the assets are downloaded into the image. If given,
    assets are downloaded to this directory and bind-mounted into the container so they can be
    reused across benchmark runs.
    :return: A callable that executes the benchmark, or `None` if `benchmark_name` is not
    supported by this executor.
    """

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
            name=benchmark_name,
            benchmark_version=version,
            override_parameters={},
            larger_better=True,
            verbose_unit=verbose_unit,
            unit=unit,
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.native_multi_gpu_result,
            numerical_results=docker_wrapper.benchmark_dockerfile(
                dockerfile_path=LLAMA_BENCH_DOCKERFILE,
                benchmark_name=benchmark_name.value,
                benchmark_version=version,
                gpus=gpus,
                create_runtime_env_vars=lambda runtime_gpus: [
                    (
                        "MODEL_SPECS",
                        str(
                            json.dumps(
                                [spec.model_dump() for spec in llama_bench_parameters.model_specs]
                            )
                        ),
                    ),
                    ("NUM_PROMPT_TOKENS", str(llama_bench_parameters.prompt_tokens)),
                    ("NUM_GENERATION_TOKENS", str(llama_bench_parameters.generation_tokens)),
                    ("NUM_REPETITIONS", str(llama_bench_parameters.repetitions)),
                    ("CONTEXT_DEPTH", str(llama_bench_parameters.context_depth)),
                ],
                multi_gpu_native=multi_gpu_native,
                outputs_to_result=outputs_to_results,
                docker_cleanup=docker_cleanup,
                force_rebuild=force_rebuild,
                downloads_directory_path=downloads_directory_path,
            ),
        )

    return run_llama_bench_docker


def create_llama_bench_throughput_executor(
    benchmark_name: BenchmarkName,
    gpus: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some llama bench benchmarks.
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

    return _abstract_llama_bench_executor(
        benchmark_name=benchmark_name,
        version=_LLAMA_BENCH_PERFORMANCE_VERSION,
        name_to_parameters={
            # --- Qwen 2.5 1.5B (Dense) ---
            BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_1_5B_MODEL],
                prompt_tokens=_NUM_TEST_TOKENS,
                generation_tokens=0,
                context_depth=0,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_1_5B_MODEL],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=0,
                repetitions=_NUM_REPETITIONS,
            ),
            # --- Meta Llama 3 8B (Dense) ---
            BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt: _LlamaBenchParams(
                model_specs=[_META_LLAMA_3_8B_MODEL],
                prompt_tokens=_NUM_TEST_TOKENS,
                generation_tokens=0,
                context_depth=0,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation: _LlamaBenchParams(
                model_specs=[_META_LLAMA_3_8B_MODEL],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=0,
                repetitions=_NUM_REPETITIONS,
            ),
            # --- Qwen 1.5 MoE A2.7B (Sparse MoE) ---
            BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt: _LlamaBenchParams(
                model_specs=[_QWEN_1_5_MOE_MODEL],
                prompt_tokens=_NUM_TEST_TOKENS,
                generation_tokens=0,
                context_depth=0,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation: _LlamaBenchParams(
                model_specs=[_QWEN_1_5_MOE_MODEL],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=0,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_32k_ctx_prompt: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_32B_Q4],
                prompt_tokens=_NUM_TEST_TOKENS,
                generation_tokens=0,
                context_depth=_CONTEXT_DEPTH_32K,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_32k_ctx_generation: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_32B_Q4],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=_CONTEXT_DEPTH_32K,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_64k_ctx_prompt: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_32B_Q4],
                prompt_tokens=_NUM_TEST_TOKENS,
                generation_tokens=0,
                context_depth=_CONTEXT_DEPTH_32K * 2,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_64k_ctx_generation: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_32B_Q4],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=_CONTEXT_DEPTH_32K * 2,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_128k_ctx_prompt: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_32B_Q4],
                prompt_tokens=_NUM_TEST_TOKENS,
                generation_tokens=0,
                context_depth=_CONTEXT_DEPTH_32K * 4,
                repetitions=_NUM_REPETITIONS,
            ),
            BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_128k_ctx_generation: _LlamaBenchParams(
                model_specs=[_QWEN_2_5_32B_Q4],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=_CONTEXT_DEPTH_32K * 4,
                repetitions=_NUM_REPETITIONS,
            ),
        },
        verbose_unit="Tokens / Second",
        unit="toks/s",
        outputs_to_results=lambda container_outputs: _parse_docker_logs(
            container_outputs
        ).tokens_per_second,
        gpus=gpus,
        docker_cleanup=docker_cleanup,
        force_rebuild=force_rebuild,
        downloads_directory_path=downloads_directory_path,
    )


def create_llama_bench_max_model_parameters_executor(
    benchmark_name: BenchmarkName,
    gpus: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> Optional[BenchmarkExecutor]:
    """
    Creates an executor that uses docker to run some llama bench benchmarks.
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

    def make_params(model_specs: List[_HFModelSpec], context_depth: int) -> _LlamaBenchParams:
        """
        Helper function to build the output type.
        :param model_specs: Models to run.
        :param context_depth: Number of params to fit in the cache.
        :return: NT
        """

        return _LlamaBenchParams(
            model_specs=model_specs,
            prompt_tokens=1,
            generation_tokens=1,
            context_depth=context_depth,
            repetitions=1,
        )

    return _abstract_llama_bench_executor(
        benchmark_name=benchmark_name,
        version=_LLAMA_BENCH_MAX_MODEL_PARAMETERS_VERSION,
        name_to_parameters={
            BenchmarkName.llama_bench_max_model_parameters_q4_k_m_4k_ctx: make_params(
                model_specs=_Q4_SPECS, context_depth=_CONTEXT_DEPTH_4K
            ),
            BenchmarkName.llama_bench_max_model_parameters_q4_k_m_32k_ctx: make_params(
                model_specs=_Q4_SPECS, context_depth=_CONTEXT_DEPTH_32K
            ),
            BenchmarkName.llama_bench_max_model_parameters_q8_4k_ctx: make_params(
                model_specs=_Q8_SPECS, context_depth=_CONTEXT_DEPTH_4K
            ),
            BenchmarkName.llama_bench_max_model_parameters_q8_32k_ctx: make_params(
                model_specs=_Q8_SPECS, context_depth=_CONTEXT_DEPTH_32K
            ),
            BenchmarkName.llama_bench_max_model_parameters_f16_4k_ctx: make_params(
                model_specs=_F16_SPECS, context_depth=_CONTEXT_DEPTH_4K
            ),
            BenchmarkName.llama_bench_max_model_parameters_f16_32k_ctx: make_params(
                model_specs=_F16_SPECS, context_depth=_CONTEXT_DEPTH_32K
            ),
        },
        verbose_unit="Loaded Parameters",
        unit="params",
        outputs_to_results=lambda c_o: _parse_docker_logs(c_o).model_number_parameters,
        gpus=gpus,
        docker_cleanup=docker_cleanup,
        force_rebuild=force_rebuild,
        downloads_directory_path=downloads_directory_path,
    )
