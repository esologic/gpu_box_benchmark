"""
Abstractions to define different named benchmarks and how they get run.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Protocol, Tuple

from pydantic import BaseModel

from gpu_box_benchmark.locate_describe_hardware import CPUIdentity, GPUIdentity


class BenchmarkFamily(str, Enum):
    """
    Used for aggregating docs.
    """

    resnet50 = "resnet50"
    llama_bench = "llama_bench"
    llama_bench_max_model_parameters = "llama_bench_max_model_parameters"
    blender_benchmark = "blender_benchmark"
    fah_bench = "fah_bench"
    ai_benchmark = "ai_benchmark"
    whisper = "whisper"
    content_aware_timelapse = "content_aware_timelapse"
    gdsio = "gdsio"
    hashcat = "hashcat"


class BenchmarkName(str, Enum):
    """
    The different benchmarks the user can choose from.
    """

    resnet50_train_batch_1 = "resnet50_train_batch_1"
    resnet50_train_batch_64 = "resnet50_train_batch_64"
    resnet50_infer_batch_1 = "resnet50_infer_batch_1"
    resnet50_infer_batch_256 = "resnet50_infer_batch_256"

    llama_bench_qwen_2_5_1_5b_instruct_prompt = "llama_bench_qwen_2_5_1_5b_instruct_prompt"
    llama_bench_qwen_2_5_1_5b_instruct_generation = "llama_bench_qwen_2_5_1_5b_instruct_generation"

    llama_bench_meta_llama_3_8b_instruct_prompt = "llama_bench_meta_llama_3_8b_instruct_prompt"
    llama_bench_meta_llama_3_8b_instruct_generation = (
        "llama_bench_meta_llama_3_8b_instruct_generation"
    )

    llama_bench_qwen_1_5_moe_chat_prompt = "llama_bench_qwen_1_5_moe_chat_prompt"
    llama_bench_qwen_1_5_moe_chat_generation = "llama_bench_qwen_1_5_moe_chat_generation"

    llama_bench_qwen_2_5_14b_q4_k_m_16k_ctx_generation = (
        "llama_bench_qwen_2_5_14b_q4_k_m_16k_ctx_generation"
    )
    llama_bench_qwen_2_5_14b_q4_k_m_32k_ctx_generation = (
        "llama_bench_qwen_2_5_14b_q4_k_m_32k_ctx_generation"
    )
    llama_bench_qwen_2_5_14b_q4_k_m_64k_ctx_generation = (
        "llama_bench_qwen_2_5_14b_q4_k_m_64k_ctx_generation"
    )
    llama_bench_qwen_2_5_14b_q4_k_m_128k_ctx_generation = (
        "llama_bench_qwen_2_5_14b_q4_k_m_128k_ctx_generation"
    )

    llama_bench_qwen_2_5_32b_q4_k_m_16k_ctx_generation = (
        "llama_bench_qwen_2_5_32b_q4_k_m_16k_ctx_generation"
    )
    llama_bench_qwen_2_5_32b_q4_k_m_32k_ctx_generation = (
        "llama_bench_qwen_2_5_32b_q4_k_m_32k_ctx_generation"
    )
    llama_bench_qwen_2_5_32b_q4_k_m_64k_ctx_generation = (
        "llama_bench_qwen_2_5_32b_q4_k_m_64k_ctx_generation"
    )
    # Doesn't work on 3x P100
    llama_bench_qwen_2_5_32b_q4_k_m_128k_ctx_generation = (
        "llama_bench_qwen_2_5_32b_q4_k_m_128k_ctx_generation"
    )

    # Deepseek Coder V2 Lite is 16B Parameters
    llama_bench_deepseek_coder_v2_lite_q8_16k_ctx_generation = (
        "llama_bench_deepseek_coder_v2_lite_q8_16k_ctx_generation"
    )
    llama_bench_deepseek_coder_v2_lite_q8_32k_ctx_generation = (
        "llama_bench_deepseek_coder_v2_lite_q8_32k_ctx_generation"
    )
    llama_bench_deepseek_coder_v2_lite_q8_64k_ctx_generation = (
        "llama_bench_deepseek_coder_v2_lite_q8_64k_ctx_generation"
    )
    llama_bench_deepseek_coder_v2_lite_q8_128k_ctx_generation = (
        "llama_bench_deepseek_coder_v2_lite_q8_128k_ctx_generation"
    )

    llama_bench_gemma_4_31b_q8_16k_ctx_generation = "llama_bench_gemma_4_31b_q8_16k_ctx_generation"
    llama_bench_gemma_4_31b_q8_32k_ctx_generation = "llama_bench_gemma_4_31b_q8_32k_ctx_generation"
    llama_bench_gemma_4_31b_q8_64k_ctx_generation = "llama_bench_gemma_4_31b_q8_64k_ctx_generation"
    # Doesn't work on 3x P100
    llama_bench_gemma_4_31b_q8_128k_ctx_generation = (
        "llama_bench_gemma_4_31b_q8_128k_ctx_generation"
    )

    llama_bench_qwen_3_6_27b_f16_4k_ctx_generation = (
        "llama_bench_qwen_3_6_27b_f16_4k_ctx_generation"
    )
    llama_bench_qwen_3_6_27b_f16_8k_ctx_generation = (
        "llama_bench_qwen_3_6_27b_f16_8k_ctx_generation"
    )
    llama_bench_qwen_3_6_27b_f16_16k_ctx_generation = (
        "llama_bench_qwen_3_6_27b_f16_16k_ctx_generation"
    )
    llama_bench_qwen_3_6_27b_f16_32k_ctx_generation = (
        "llama_bench_qwen_3_6_27b_f16_32k_ctx_generation"
    )
    llama_bench_qwen_3_6_27b_f16_64k_ctx_generation = (
        "llama_bench_qwen_3_6_27b_f16_64k_ctx_generation"
    )
    llama_bench_qwen_3_6_27b_f16_128k_ctx_generation = (
        "llama_bench_qwen_3_6_27b_f16_128k_ctx_generation"
    )

    llama_bench_gemma_4_12b_f16_4k_ctx_generation = "llama_bench_gemma_4_12b_f16_4k_ctx_generation"
    llama_bench_gemma_4_12b_f16_8k_ctx_generation = "llama_bench_gemma_4_12b_f16_8k_ctx_generation"
    llama_bench_gemma_4_12b_f16_16k_ctx_generation = (
        "llama_bench_gemma_4_12b_f16_16k_ctx_generation"
    )
    llama_bench_gemma_4_12b_f16_32k_ctx_generation = (
        "llama_bench_gemma_4_12b_f16_32k_ctx_generation"
    )
    llama_bench_gemma_4_12b_f16_64k_ctx_generation = (
        "llama_bench_gemma_4_12b_f16_64k_ctx_generation"
    )
    llama_bench_gemma_4_12b_f16_128k_ctx_generation = (
        "llama_bench_gemma_4_12b_f16_128k_ctx_generation"
    )

    llama_bench_max_model_parameters_q4_k_m_4k_ctx = (
        "llama_bench_max_model_parameters_q4_k_m_4k_ctx"
    )
    llama_bench_max_model_parameters_q4_k_m_32k_ctx = (
        "llama_bench_max_model_parameters_q4_k_m_32k_ctx"
    )
    llama_bench_max_model_parameters_q8_4k_ctx = "llama_bench_max_model_parameters_q8_4k_ctx"
    llama_bench_max_model_parameters_q8_32k_ctx = "llama_bench_max_model_parameters_q8_32k_ctx"
    llama_bench_max_model_parameters_f16_4k_ctx = "llama_bench_max_model_parameters_f16_4k_ctx"
    llama_bench_max_model_parameters_f16_32k_ctx = "llama_bench_max_model_parameters_f16_32k_ctx"

    blender_benchmark_monster_cpu = "blender_benchmark_monster_cpu"
    blender_benchmark_monster_gpu = "blender_benchmark_monster_gpu"

    fah_bench_single = "fah_bench_single"
    fah_bench_double = "fah_bench_double"

    ai_benchmark = "ai_benchmark"

    whisper_medium_fp16 = "whisper_medium_fp16"

    content_aware_timelapse_vit_scores = "content_aware_timelapse_vit_scores"
    content_aware_timelapse_vit_attention = "content_aware_timelapse_vit_attention"

    gdsio_type_0 = "gdsio_type_0"
    gdsio_type_2 = "gdsio_type_2"

    hashcat_sha256 = "hashcat_sha256"


BENCHMARK_TO_PRETTY: Dict[BenchmarkName, str] = {
    # ResNet Benchmarks
    BenchmarkName.resnet50_train_batch_1: "ResNet50 Train B=1",
    BenchmarkName.resnet50_train_batch_64: "ResNet50 Train B=64",
    BenchmarkName.resnet50_infer_batch_1: "ResNet50 Infer B=1",
    BenchmarkName.resnet50_infer_batch_256: "ResNet50 Infer B=256",
    # Qwen 2.5 1.5B
    BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt: ("llama.cpp Qwen2.5 1.5B Prompt"),
    BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation: ("llama.cpp Qwen2.5 1.5B Gen"),
    # Llama 3 8B
    BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt: ("llama.cpp Llama 3 8B Prompt"),
    BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation: ("llama.cpp Llama 3 8B Gen"),
    # Qwen 1.5 MoE
    BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt: ("llama.cpp Qwen1.5 MoE Prompt"),
    BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation: ("llama.cpp Qwen1.5 MoE Gen"),
    # Qwen 2.5 14B Q4_K_M
    BenchmarkName.llama_bench_qwen_2_5_14b_q4_k_m_16k_ctx_generation: (
        "llama.cpp Qwen2.5 14B Q4_K_M Gen (16K)"
    ),
    BenchmarkName.llama_bench_qwen_2_5_14b_q4_k_m_32k_ctx_generation: (
        "llama.cpp Qwen2.5 14B Q4_K_M Gen (32K)"
    ),
    BenchmarkName.llama_bench_qwen_2_5_14b_q4_k_m_64k_ctx_generation: (
        "llama.cpp Qwen2.5 14B Q4_K_M Gen (64K)"
    ),
    BenchmarkName.llama_bench_qwen_2_5_14b_q4_k_m_128k_ctx_generation: (
        "llama.cpp Qwen2.5 14B Q4_K_M Gen (128K)"
    ),
    # Qwen 2.5 32B Q4_K_M
    BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_16k_ctx_generation: (
        "llama.cpp Qwen2.5 32B Q4_K_M Gen (16K)"
    ),
    BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_32k_ctx_generation: (
        "llama.cpp Qwen2.5 32B Q4_K_M Gen (32K)"
    ),
    BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_64k_ctx_generation: (
        "llama.cpp Qwen2.5 32B Q4_K_M Gen (64K)"
    ),
    BenchmarkName.llama_bench_qwen_2_5_32b_q4_k_m_128k_ctx_generation: (
        "llama.cpp Qwen2.5 32B Q4_K_M Gen (128K)"
    ),
    # Gemma 4 31B Q8
    BenchmarkName.llama_bench_gemma_4_31b_q8_16k_ctx_generation: (
        "llama.cpp Gemma 4 31B Q8 Gen (16K)"
    ),
    BenchmarkName.llama_bench_gemma_4_31b_q8_32k_ctx_generation: (
        "llama.cpp Gemma 4 31B Q8 Gen (32K)"
    ),
    BenchmarkName.llama_bench_gemma_4_31b_q8_64k_ctx_generation: (
        "llama.cpp Gemma 4 31B Q8 Gen (64K)"
    ),
    BenchmarkName.llama_bench_gemma_4_31b_q8_128k_ctx_generation: (
        "llama.cpp Gemma 4 31B Q8 Gen (128K)"
    ),
    # Qwen 3.6 27B F16
    BenchmarkName.llama_bench_qwen_3_6_27b_f16_4k_ctx_generation: (
        "llama.cpp Qwen3.6 27B F16 Gen (4K)"
    ),
    BenchmarkName.llama_bench_qwen_3_6_27b_f16_8k_ctx_generation: (
        "llama.cpp Qwen3.6 27B F16 Gen (8K)"
    ),
    BenchmarkName.llama_bench_qwen_3_6_27b_f16_16k_ctx_generation: (
        "llama.cpp Qwen3.6 27B F16 Gen (16K)"
    ),
    BenchmarkName.llama_bench_qwen_3_6_27b_f16_32k_ctx_generation: (
        "llama.cpp Qwen3.6 27B F16 Gen (32K)"
    ),
    BenchmarkName.llama_bench_qwen_3_6_27b_f16_64k_ctx_generation: (
        "llama.cpp Qwen3.6 27B F16 Gen (64K)"
    ),
    BenchmarkName.llama_bench_qwen_3_6_27b_f16_128k_ctx_generation: (
        "llama.cpp Qwen3.6 27B F16 Gen (128K)"
    ),
    # Gemma 4 12B F16
    BenchmarkName.llama_bench_gemma_4_12b_f16_4k_ctx_generation: (
        "llama.cpp Gemma 4 12B F16 Gen (4K)"
    ),
    BenchmarkName.llama_bench_gemma_4_12b_f16_8k_ctx_generation: (
        "llama.cpp Gemma 4 12B F16 Gen (8K)"
    ),
    BenchmarkName.llama_bench_gemma_4_12b_f16_16k_ctx_generation: (
        "llama.cpp Gemma 4 12B F16 Gen (16K)"
    ),
    BenchmarkName.llama_bench_gemma_4_12b_f16_32k_ctx_generation: (
        "llama.cpp Gemma 4 12B F16 Gen (32K)"
    ),
    BenchmarkName.llama_bench_gemma_4_12b_f16_64k_ctx_generation: (
        "llama.cpp Gemma 4 12B F16 Gen (64K)"
    ),
    BenchmarkName.llama_bench_gemma_4_12b_f16_128k_ctx_generation: (
        "llama.cpp Gemma 4 12B F16 Gen (128K)"
    ),
    # Maximum Model Size
    BenchmarkName.llama_bench_max_model_parameters_q4_k_m_4k_ctx: (
        "llama.cpp Max Model Parameters (Q4_K_M) + 4K Context"
    ),
    BenchmarkName.llama_bench_max_model_parameters_q4_k_m_32k_ctx: (
        "llama.cpp Max Model Parameters (Q4_K_M) + 32K Context"
    ),
    BenchmarkName.llama_bench_max_model_parameters_q8_4k_ctx: (
        "llama.cpp Max Model Parameters (Q8) + 4K Context"
    ),
    BenchmarkName.llama_bench_max_model_parameters_q8_32k_ctx: (
        "llama.cpp Max Model Parameters (Q8) + 32K Context"
    ),
    BenchmarkName.llama_bench_max_model_parameters_f16_4k_ctx: (
        "llama.cpp Max Model Parameters (F16) + 4K Context"
    ),
    BenchmarkName.llama_bench_max_model_parameters_f16_32k_ctx: (
        "llama.cpp Max Model Parameters (F16) + 32K Context"
    ),
    # Hardware & Other Benchmarks
    BenchmarkName.blender_benchmark_monster_cpu: "Blender CPU",
    BenchmarkName.blender_benchmark_monster_gpu: "Blender GPU",
    BenchmarkName.fah_bench_single: "F@H Single",
    BenchmarkName.fah_bench_double: "F@H Double",
    BenchmarkName.ai_benchmark: "AI Benchmark",
    BenchmarkName.whisper_medium_fp16: "Whisper Med FP16",
    BenchmarkName.content_aware_timelapse_vit_scores: "CAT ViT Scores",
    BenchmarkName.content_aware_timelapse_vit_attention: "CAT ViT Attention",
    BenchmarkName.gdsio_type_0: "Storage→GPU (GDSIO)",
    BenchmarkName.gdsio_type_2: "Storage→CPU→GPU (GDSIO)",
    BenchmarkName.hashcat_sha256: "SHA-256",
}

EXTENDED_BENCHMARK_DOCUMENTS: Dict[BenchmarkFamily, str] = {
    BenchmarkFamily.resnet50: (
        "From the NVidia Deep Learning Examples Repo, the ResNet50 benchmark uses the pytorch "
        "backend to run a workload on the GPU. The benchmark uses a synthetic data backend, so it "
        "isolates raw compute and framework performance without being limited by disk or "
        "data-loading I/O. t can be configured via environment variables to measure either "
        "training or inference performance."
    ),
    BenchmarkFamily.llama_bench: (
        "This benchmark uses the CUDA-enabled llama.cpp container to measure large language model "
        "inference performance on the GPU using the purpose-built llama-bench tool. The container "
        "downloads requested models at runtime. Stats are recorded as tokens/second."
    ),
    BenchmarkFamily.llama_bench_max_model_parameters: (
        "Similar to the other llama bench set of benchmarks, but this set of tests looks to see "
        "the largest model that can fit into the attached GPUs with a set context. "
    ),
    BenchmarkFamily.blender_benchmark: (
        "This benchmark uses Blender’s official Open Data benchmark suite to measure GPU rendering "
        "performance in a standardized, real-world workload. The workload exercises GPU compute, "
        "memory, and driver performance Overall, this benchmark evaluates how well a GPU performs "
        "on production-style 3D rendering tasks that closely reflect professional content-creation "
        "use cases. This is not an AI-related benchmark."
    ),
    BenchmarkFamily.fah_bench: (
        "This benchmark builds and runs FAHBench, the Folding@home microbenchmark suite, to "
        "evaluate GPU compute performance using OpenCL. The container compiles FAHBench from "
        "source with the GUI disabled, ensuring a headless, reproducible setup suitable for "
        "automated benchmarking.The workload stresses floating-point throughput, memory access "
        "patterns, and driver stability in a scientific computing context rather than graphics or "
        "deep learning. Overall, this benchmark measures how well a GPU performs on sustained, "
        "real-world molecular dynamics–style calculations similar to Folding@home workloads."
    ),
    BenchmarkFamily.ai_benchmark: (
        "This benchmark runs the AI Benchmark suite to evaluate end-to-end deep learning "
        "performance on the GPU using TensorFlow. Overall, it produces a single composite AI "
        "score that provides a high-level comparison of how well a GPU performs across common "
        "deep learning tasks."
    ),
    BenchmarkFamily.whisper: (
        "This benchmark measures GPU-accelerated speech-to-text performance using OpenAI’s "
        "Whisper “medium” model running on PyTorch with CUDA. At runtime, a fixed audio sample "
        "is transcribed on the GPU, ensuring consistent input across benchmark runs. The script "
        "measures only the transcription phase and reports throughput as mel-spectrogram frames "
        "processed per second. Overall, this test evaluates real-world GPU inference performance "
        "for transformer-based audio models rather than synthetic or microbenchmark workloads."
    ),
    BenchmarkFamily.content_aware_timelapse: (
        "Uses benchmark mode in content aware timelapse to measure throughput of a GPU or multiple"
        "GPUs through a ViT model. Has both a fused (score) and slower unused (attention) modes. "
        "This series of benchmarks is very relevant to the author if this repo as it is why the "
        "development is happening in the first place."
    ),
    BenchmarkFamily.gdsio: (
        "This benchmark uses NVIDIA’s gdsio utility to measure the raw data transfer throughput "
        "between storage and GPU memory. By utilizing NVIDIA GPUDirect Storage (GDS) tools in "
        "compatibility mode, it simulates the heavy I/O workload of loading large model weights "
        "from disk into VRAM or offloading KV-caches. Overall, this test isolates the "
        "storage-to-GPU pipeline, providing a critical metric for understanding model-load "
        "latencies and multi-GPU data orchestration performance."
    ),
    BenchmarkFamily.hashcat: (
        "This benchmark uses Hashcat’s built-in benchmark mode to measure raw cryptographic "
        "hashing throughput on the GPU. It runs a selected fast, unsalted hash "
        "algorithm (such as SHA-256) in a tight compute loop, providing a synthetic but highly "
        "GPU-bound workload that closely reflects the arithmetic intensity of proof-of-work style "
        "hashing."
    ),
}
"""
Used in docs.
"""


class NumericalResultKey(str, Enum):
    """
    For programmatically looking up values in ReportFileNumerical.
    """

    min_by_gpu_type = "min_by_gpu_type"
    max_by_gpu_type = "max_by_gpu_type"
    mean_by_gpu_type = "mean_by_gpu_type"

    theoretical_multi_gpu_mean = "theoretical_multi_gpu_mean"
    theoretical_multi_gpu_sum = "theoretical_multi_gpu_sum"

    forced_multi_gpu_numerical_mean = "forced_multi_gpu_numerical_mean"
    forced_multi_gpu_sum = "forced_multi_gpu_sum"

    native_multi_gpu_result = "native_multi_gpu_result"


class ReportFileNumerical(BaseModel):
    """
    Contain the numerical results for the run. Units etc. are kept in the outer result.

    The keys: min_by_gpu_type, max_by_gpu_type, mean_by_gpu_type, theoretical_multi_gpu_mean,
    theoretical_multi_gpu_sum, forced_multi_gpu_numerical_mean, forced_multi_gpu_sum are all
    related to single GPU performance. If a benchmark fails on ANY of the individual GPUs involved,
    these keys get set to None.
    """

    # Single GPU performance results
    min_by_gpu_type: Optional[float]
    max_by_gpu_type: Optional[float]
    mean_by_gpu_type: Optional[float]
    theoretical_multi_gpu_mean: Optional[float]
    theoretical_multi_gpu_sum: Optional[float]
    forced_multi_gpu_numerical_mean: Optional[float]
    forced_multi_gpu_sum: Optional[float]

    # True multi GPU performance results
    native_multi_gpu_result: Optional[float]
    """
    Some benchmarks do not natively support multi GPU utilization. In those cases where it is not
    supported, this will be set to None.
    
    This could either map onto the theoretical sum _or_ mean depending on the benchmark. If it 
    becomes hard to distinguish here, add a a flag to the outer type. None here means there's no
    native multi-gpu scoring supported.
    """


class BenchmarkResult(BaseModel):
    """
    Describes an individual benchmark result. This should be the output of a single test.
    """

    name: BenchmarkName
    """
    The test that was run in this evaluation.
    """

    benchmark_version: str
    """
    Version of the benchmark that was run, not the version of the entire suite. This way benchmarks
    can be versioned independent to each-other. 
    """

    override_parameters: Dict[str, str | float | int | bool]
    """
    Records excursions from the default set of parameters for the given benchmark. Make a new test
    if this is being used often. 
    """

    larger_better: bool
    """
    Flag on how to interpret results vs other runs. 
    """

    multi_gpu_native: bool

    verbose_unit: str
    unit: str

    critical_result_key: NumericalResultKey
    """
    Points to the value in `numerical_results` that is the highlight value for this test.
    Makes comparison a bit easier while keeping all parts of the results. 
    """

    numerical_results: ReportFileNumerical


class BenchmarkExecutor(Protocol):
    """
    Defines the callables that run to produce the benchmark results.
    """

    def __call__(self) -> BenchmarkResult:
        """
        Takes no arguments and produces the benchmark results.
        :return: The filled benchmark results.
        """


class CreateBenchmarkExecutor(Protocol):
    """
    Benchmark wrapper modules expose these functions for going from a user desired benchmark name
    to the pre-configured function to actually execute the function.
    """

    def __call__(  # pylint: disable=too-many-positional-arguments
        self,
        benchmark_name: BenchmarkName,
        gpus: Tuple[GPUIdentity, ...],
        docker_cleanup: bool,
        force_rebuild: bool,
        downloads_directory_path: Optional[Path],
    ) -> Optional[BenchmarkExecutor]:
        """
        :param benchmark_name: The enum representation of the benchmark to run.
        :param gpus: List of GPUs to run the benchmark on. Jobs can utilize the GPU or decide not
        to use them.
        :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
        :param force_rebuild: If given, the underlying docker containers are rebuilt from scratch
        before running. This will take more time that using the existing images.
        :param downloads_directory_path: Benchmark docker containers may need to download test
        assets during build time or runtime. By default, if this is not given, those assets are
        downloaded into the image. If this is given, the assets are downloaded to this location
        which is bind
        mounted into the container so the downloads can be re-used.
        :return: Returns None if the wrapper module does not contain the desired benchmark.
        """


class SystemEvaluation(BaseModel):
    """
    Describes a series of benchmarks and the system that ran them.

    You'll notice there is nothing about the stats (CPU speed, memory speed, GPU speed etc) in the
    system description section. The idea here is that these should be evaluated with benchmarks vs.
    recording the theoretical max speed of the CPU for example.
    """

    title: str
    description: str
    gpu_box_benchmark_version: str
    """
    Version of the entire suite of benchmarks. 
    """

    start_time: datetime
    runtime_seconds: float

    cpu: CPUIdentity

    total_memory_gb: float

    gpus: Tuple[GPUIdentity, ...]

    results: List[BenchmarkResult]


def load_system_evaluation_from_disk(path: Path) -> SystemEvaluation:
    """
    Loads a system evaluation json file from disk into it's in-memory representation.
    :param path: Path to the file.
    :return: Loaded model.
    """

    with path.open() as file:
        return SystemEvaluation.model_validate_json(file.read())


class NamedExecutor(NamedTuple):
    """
    Intermediate Type for labeling executors.
    """

    benchmark_name: BenchmarkName
    executor: BenchmarkExecutor
