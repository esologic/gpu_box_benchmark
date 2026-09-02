"""
Type code specific to working with LLMs
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple

from pydantic import BaseModel

from gpu_box_benchmark.benchmark_types import BenchmarkName


class HFModelSpec(BaseModel):
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


class ModelQuantization(NamedTuple):
    """
    Locations of the different quantization's of a given model.
    """

    q4: HFModelSpec
    q8: HFModelSpec
    f16: HFModelSpec


class LlamaBenchParams(NamedTuple):
    """
    Defines the knobs that can be turned for a ResNet50 Benchmark run.
    """

    model_specs: List[HFModelSpec]
    prompt_tokens: int
    generation_tokens: int
    context_depth: int
    repetitions: int


class LlamaBenchOutputs(NamedTuple):
    """
    Different outputs from llama.cpp that we care about from the runs.
    """

    tokens_per_second: float
    model_number_parameters: int


class LLMName(str, Enum):
    """
    Canonical names for the different supported models.
    """

    qwen_2_5_3b_instruct = "Qwen 2.5 3B Instruct"
    qwen_2_5_14b_instruct = "Qwen 2.5 14B Instruct"
    qwen_2_5_32b_instruct = "Qwen 2.5 32B Instruct"
    qwen_2_5_72b_instruct = "Qwen 2.5 72B Instruct"
    deepseek_coder_v2_lite = "DeepSeek Coder V2 Lite"
    gemma_4_8b = "Gemma 4 8B"
    gemma_4_12b = "Gemma 4 12B"
    gemma_4_26b = "Gemma 4 26B"
    gemma_4_31b = "Gemma 4 31B"
    qwen_3_6_27b = "Qwen 3.6 27B"
    qwen_3_6_35b = "Qwen 3.6 35B"


@dataclass(frozen=True)
class LlamaBenchmarkDefinition:
    """
    Defines the configuration and metadata for a single llama.cpp benchmark.
    """

    benchmark_name: BenchmarkName
    model: LLMName
    quantization: str
    context_depth: int
    model_spec: HFModelSpec


@dataclass(frozen=True)
class LLMThroughputResult:
    """
    Downstream from a ReportFileNumerical, consumed by visualization code.
    """

    label: str
    """
    Identifies the run that produced the result. This is used to group and compare results from
    the same system or software configuration.
    """

    model: LLMName
    """
    Model being benchmarked.
    """

    quantization: str
    """
    Quantization format used for the model.
    """

    context_depth: int
    """
    Context depth used for the benchmark, measured in tokens.
    """

    tokens_per_second: float
    """
    Measured generation throughput, in tokens per second.
    """


CONTEXT_DEPTH_4K = 4096
CONTEXT_DEPTH_16K = 16384
CONTEXT_DEPTH_32K = 32768
