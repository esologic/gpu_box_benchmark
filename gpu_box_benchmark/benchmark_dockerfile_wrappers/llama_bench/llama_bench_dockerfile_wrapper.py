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
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import HfApi

from benchmark_dockerfiles import LLAMA_BENCH_DOCKERFILE
from gpu_box_benchmark import docker_wrapper
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench import hf_models
from gpu_box_benchmark.benchmark_dockerfile_wrappers.llama_bench.llama_bench_types import (
    CONTEXT_DEPTH_4K,
    CONTEXT_DEPTH_32K,
    HFModelSpec,
    LlamaBenchmarkDefinition,
    LlamaBenchOutputs,
    LlamaBenchParams,
    LLMName,
    LLMThroughputResult,
    ModelQuantization,
)
from gpu_box_benchmark.benchmark_types import (
    BenchmarkExecutor,
    BenchmarkName,
    BenchmarkResult,
    GPUPowerUsageKey,
    NumericalResultKey,
)
from gpu_box_benchmark.docker_wrapper import ContainerOutputs, OutputsToResult
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)

_LLAMA_BENCH_PERFORMANCE_VERSION = "0.4.0"
"""
The benchmarks related to Tok/S throughput.

# Version History

## 0.4.0 - (2026-08-XX)
* Updated runner script to optionally run multiple models to enable the max parameters benchmark.
* Touched a bunch of code to decrease redundancy between the two testing modalities.
* Added tests for a set of new models, quants, and context lengths per feedback from first release.
* Added --no-host and --no-kv-offload=false to force usage of GPU resources over system CPU/RAM. 

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

## 0.1.0 - (2026-08-XX)
* First version 
"""

_NUM_TEST_TOKENS = 512
_NUM_REPETITIONS = 10


def _parse_docker_logs(container_outputs: ContainerOutputs) -> LlamaBenchOutputs:
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

    return LlamaBenchOutputs(
        tokens_per_second=float(summary_dict["mean"]),
        model_number_parameters=loaded["model_n_params"],
    )


def verify_hf_model_specs(model_specs: List[HFModelSpec]) -> None:
    """
    Verifies that all model specs actually exist on Hugging Face.

    :param model_specs: To verify.
    :return: None
    :raises RuntimeError: If a repo is invalid or an entrypoint file is missing.
    """
    api = HfApi()

    for spec in model_specs:
        # Check if the repository exists and is accessible
        try:
            # Fetching files validates access and network connectivity
            files = api.list_repo_files(repo_id=spec.repo)
        except Exception as e:
            raise RuntimeError(f"Failed to verify repository '{spec.repo}': {e}") from e

        # Verify the entrypoint file exists in that repo
        if spec.entrypoint not in files:
            # Suggest similar filenames to help debugging
            potential_matches = [f for f in files if spec.entrypoint in f]
            msg = (
                f"Model entrypoint '{spec.entrypoint}' does not exist in "
                f"repository '{spec.repo}'."
            )
            if potential_matches:
                msg += "\nDid you mean one of the following?\n" + "\n".join(
                    [f"  - {m}" for m in potential_matches]
                )
            raise RuntimeError(msg)


def _abstract_llama_bench_executor(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    benchmark_name: BenchmarkName,
    version: str,
    name_to_parameters: Dict[BenchmarkName, LlamaBenchParams],
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

    llama_bench_parameters: Optional[LlamaBenchParams] = name_to_parameters.get(
        benchmark_name, None
    )

    if llama_bench_parameters is None:
        return None

    # Raises an exception if the given model/models don't exist on HF.
    verify_hf_model_specs(llama_bench_parameters.model_specs)

    def run_llama_bench_docker() -> BenchmarkResult:
        """
        Build and run the docker image that runs the benchmark.
        :return: Parsed results.
        """

        multi_gpu_native = True

        benchmark_data = docker_wrapper.benchmark_dockerfile(
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
                (
                    "TENSOR_SPLIT",
                    ",".join(
                        ["1"] * len(runtime_gpus)
                        if (len(runtime_gpus) > 1 and llama_bench_parameters.even_tensor_split)
                        else "0"
                    ),
                ),
            ],
            multi_gpu_native=multi_gpu_native,
            docker_cleanup=docker_cleanup,
            force_rebuild=force_rebuild,
            downloads_directory_path=downloads_directory_path,
            outputs_to_result=outputs_to_results,
        )

        return BenchmarkResult(
            name=benchmark_name,
            benchmark_version=version,
            override_parameters={},
            larger_better=True,
            verbose_unit=verbose_unit,
            unit=unit,
            multi_gpu_native=multi_gpu_native,
            critical_result_key=NumericalResultKey.native_multi_gpu_result,
            numerical_results=benchmark_data.report_file_numerical,
            critical_gpu_power_usage_key=GPUPowerUsageKey.native_multi_gpu_total_energy_joules,
            gpu_power_usage=benchmark_data.report_file_gpu_power_usage,
        )

    return run_llama_bench_docker


SUPPORTED_QUANTS: Tuple[str, str, str] = ("q4", "q8", "f16")

SUPPORTED_CONTEXTS: Dict[str, int] = {
    "4k": CONTEXT_DEPTH_4K,
    "32k": CONTEXT_DEPTH_32K,
    "64k": CONTEXT_DEPTH_32K * 2,
    "128k": CONTEXT_DEPTH_32K * 4,
}

_SUPPORTED_MODELS: Dict[LLMName, Tuple[str, ModelQuantization]] = {
    LLMName.qwen_2_5_3b_instruct: (
        "qwen_2_5_3b_instruct",
        hf_models.QWEN_2_5_3B_INSTRUCT,
    ),
    LLMName.qwen_2_5_14b_instruct: (
        "qwen_2_5_14b_instruct",
        hf_models.QWEN_2_5_14B_INSTRUCT,
    ),
    LLMName.qwen_2_5_32b_instruct: (
        "qwen_2_5_32b_instruct",
        hf_models.QWEN_2_5_32B_INSTRUCT,
    ),
    LLMName.qwen_2_5_72b_instruct: (
        "qwen_2_5_72b_instruct",
        hf_models.QWEN_2_5_72B_INSTRUCT,
    ),
    LLMName.gemma_4_8b: (
        "gemma_4_8b",
        hf_models.GEMMA_4_E4B_8B,
    ),
    LLMName.gemma_4_12b: (
        "gemma_4_12b",
        hf_models.GEMMA_4_12B,
    ),
    LLMName.gemma_4_26b: (
        "gemma_4_26b",
        hf_models.GEMMA_4_26B,
    ),
    LLMName.gemma_4_31b: (
        "gemma_4_31b",
        hf_models.GEMMA_4_31B,
    ),
    LLMName.qwen_3_6_27b: (
        "qwen_3_6_27b",
        hf_models.QWEN_3_6_27B,
    ),
    LLMName.qwen_3_6_35b: (
        "qwen_3_6_35b",
        hf_models.QWEN_3_6_35B,
    ),
}

_LLM_BENCHMARK_DEFINITIONS: Tuple[LlamaBenchmarkDefinition, ...] = tuple(
    LlamaBenchmarkDefinition(
        benchmark_name=getattr(
            BenchmarkName,
            f"llama_bench_{model_name}_{quantization}_{context_name}_ctx_generation",
        ),
        model=model,
        quantization=quantization,
        context_depth=context_depth,
        model_spec=getattr(model_spec, quantization),
    )
    for model, (model_name, model_spec) in _SUPPORTED_MODELS.items()
    for quantization in SUPPORTED_QUANTS
    for context_name, context_depth in SUPPORTED_CONTEXTS.items()
)

_LLM_BENCHMARK_DEFINITIONS_BY_NAME: Dict[BenchmarkName, LlamaBenchmarkDefinition] = {
    definition.benchmark_name: definition for definition in _LLM_BENCHMARK_DEFINITIONS
}


def create_llama_bench_throughput_executor(  # pylint: disable=too-many-locals
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

    # These are the classic benchmarks with no preloaded context
    name_to_parameters = {
        # --- Qwen 2.5 1.5B (Dense) ---
        BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt: LlamaBenchParams(
            model_specs=[hf_models.QWEN_2_5_1_5B_MODEL],
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
            context_depth=0,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=False,
        ),
        BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation: LlamaBenchParams(
            model_specs=[hf_models.QWEN_2_5_1_5B_MODEL],
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
            context_depth=0,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=False,
        ),
        # --- Meta Llama 3 8B (Dense) ---
        BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt: LlamaBenchParams(
            model_specs=[hf_models.META_LLAMA_3_8B_MODEL],
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
            context_depth=0,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=False,
        ),
        BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation: LlamaBenchParams(
            model_specs=[hf_models.META_LLAMA_3_8B_MODEL],
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
            context_depth=0,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=False,
        ),
        # --- Qwen 1.5 MoE A2.7B (Sparse MoE) ---
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt: LlamaBenchParams(
            model_specs=[hf_models.QWEN_1_5_MOE_MODEL],
            prompt_tokens=_NUM_TEST_TOKENS,
            generation_tokens=0,
            context_depth=0,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=False,
        ),
        BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation: LlamaBenchParams(
            model_specs=[hf_models.QWEN_1_5_MOE_MODEL],
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
            context_depth=0,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=False,
        ),
        BenchmarkName.llama_bench_gemma_4_8b_q4_4k_ctx_generation_e_tensor_split: LlamaBenchParams(
            model_specs=[hf_models.GEMMA_4_E4B_8B.q4],
            prompt_tokens=0,
            generation_tokens=_NUM_TEST_TOKENS,
            context_depth=CONTEXT_DEPTH_4K,
            repetitions=_NUM_REPETITIONS,
            even_tensor_split=True,
        ),
    }

    name_to_parameters.update(
        {
            definition.benchmark_name: LlamaBenchParams(
                model_specs=[definition.model_spec],
                prompt_tokens=0,
                generation_tokens=_NUM_TEST_TOKENS,
                context_depth=definition.context_depth,
                repetitions=_NUM_REPETITIONS,
                even_tensor_split=False,
            )
            for definition in _LLM_BENCHMARK_DEFINITIONS
        }
    )

    return _abstract_llama_bench_executor(
        benchmark_name=benchmark_name,
        version=_LLAMA_BENCH_PERFORMANCE_VERSION,
        name_to_parameters=name_to_parameters,
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


def read_throughput_results(
    complete_results: List[BenchmarkResult],
    failed_tests: List[BenchmarkName],
    label: str,
) -> List[LLMThroughputResult]:
    """
    Parses out the details from a series of LLM throughput tests into a set of structs ready
    for visualization.
    :param complete_results: Contains the LLM benchmark results. Output from a complete results
    parse.
    :param failed_tests: Failed tests in the parse.
    :param label: Used to aid visualization, added to the output.
    :return: A list of parsed NTs from the input.
    """

    parsed_results: List[LLMThroughputResult] = []

    # Process successful numerical results
    for benchmark_result in complete_results:
        name = benchmark_result.name

        success_definition: LlamaBenchmarkDefinition = _LLM_BENCHMARK_DEFINITIONS_BY_NAME.get(name)
        if success_definition is None:
            continue

        tokens_per_second = getattr(
            benchmark_result.numerical_results,
            benchmark_result.critical_result_key.value,
        )

        parsed_results.append(
            LLMThroughputResult(
                label=label,
                model=success_definition.model,
                quantization=success_definition.quantization,
                context_depth=success_definition.context_depth,
                tokens_per_second=tokens_per_second,
            )
        )

    # Process failed tests (The "missing" plots)
    # We iterate the known failures and emit a row so they appear in the data for plotting.
    seen_failed_names = set()
    for name in set(failed_tests):

        if name in seen_failed_names:
            continue

        seen_failed_names.add(name)

        failed_definition: LlamaBenchmarkDefinition = _LLM_BENCHMARK_DEFINITIONS_BY_NAME.get(name)
        if failed_definition is None:
            continue

        # Emit a row with coordinates but no metric
        parsed_results.append(
            LLMThroughputResult(
                label=label,
                model=failed_definition.model,
                quantization=failed_definition.quantization,
                context_depth=failed_definition.context_depth,
                tokens_per_second=None,  # None here translates to -> the test failed.
            )
        )

    return parsed_results


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

    def make_params(model_specs: List[HFModelSpec], context_depth: int) -> LlamaBenchParams:
        """
        Helper function to build the output type.
        :param model_specs: Models to run.
        :param context_depth: Number of params to fit in the cache.
        :return: NT
        """

        return LlamaBenchParams(
            model_specs=model_specs,
            prompt_tokens=1,
            generation_tokens=1,
            context_depth=context_depth,
            repetitions=1,
            even_tensor_split=False,
        )

    return _abstract_llama_bench_executor(
        benchmark_name=benchmark_name,
        version=_LLAMA_BENCH_MAX_MODEL_PARAMETERS_VERSION,
        name_to_parameters={
            BenchmarkName.llama_bench_max_model_parameters_q4_4k_ctx: make_params(
                model_specs=hf_models.MODEL_SIZE_Q4_SPECS, context_depth=CONTEXT_DEPTH_4K
            ),
            BenchmarkName.llama_bench_max_model_parameters_q4_32k_ctx: make_params(
                model_specs=hf_models.MODEL_SIZE_Q4_SPECS, context_depth=CONTEXT_DEPTH_32K
            ),
            BenchmarkName.llama_bench_max_model_parameters_q8_4k_ctx: make_params(
                model_specs=hf_models.MODEL_SIZE_Q8_SPECS, context_depth=CONTEXT_DEPTH_4K
            ),
            BenchmarkName.llama_bench_max_model_parameters_q8_32k_ctx: make_params(
                model_specs=hf_models.MODEL_SIZE_Q8_SPECS, context_depth=CONTEXT_DEPTH_32K
            ),
            BenchmarkName.llama_bench_max_model_parameters_f16_4k_ctx: make_params(
                model_specs=hf_models.MODEL_SIZE_F16_SPECS, context_depth=CONTEXT_DEPTH_4K
            ),
            BenchmarkName.llama_bench_max_model_parameters_f16_32k_ctx: make_params(
                model_specs=hf_models.MODEL_SIZE_F16_SPECS, context_depth=CONTEXT_DEPTH_32K
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
