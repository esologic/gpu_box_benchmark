"""
Abstractions to define different named benchmarks and how they get run.
"""

from enum import Enum
from typing import Dict, Optional, Protocol, Tuple

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import BenchmarkResult


class BenchmarkFamily(str, Enum):
    """
    Used for aggregating docs.
    """

    resnet50 = "resnet50"
    llama_bench = "llama_bench"
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
    # Qwen 2.5 1.5B (Dense)
    BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt: "llama.cpp Qwen2.5 1.5B Prompt",
    BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation: "llama.cpp Qwen2.5 1.5B Gen",
    # Llama 3 8B (Dense)
    BenchmarkName.llama_bench_meta_llama_3_8b_instruct_prompt: "llama.cpp Llama 3 8B Prompt",
    BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation: "llama.cpp Llama 3 8B Gen",
    # Qwen 1.5 MoE (Sparse)
    BenchmarkName.llama_bench_qwen_1_5_moe_chat_prompt: "llama.cpp Qwen1.5 MoE Prompt",
    BenchmarkName.llama_bench_qwen_1_5_moe_chat_generation: "llama.cpp Qwen1.5 MoE Gen",
    # Hardware & Other Benchmarks
    BenchmarkName.blender_benchmark_monster_cpu: "Blender CPU",
    BenchmarkName.blender_benchmark_monster_gpu: "Blender GPU",
    BenchmarkName.fah_bench_single: "F@H Single",
    BenchmarkName.fah_bench_double: "F@H Double",
    BenchmarkName.ai_benchmark: "AI Benchmark",
    BenchmarkName.whisper_medium_fp16: "Whisper Med FP16",
    BenchmarkName.content_aware_timelapse_vit_scores: "CAT ViT Scores",
    BenchmarkName.content_aware_timelapse_vit_attention: "CAT ViT Attention",
    BenchmarkName.gdsio_type_0: "Storage->GPU (gdiso)",
    BenchmarkName.gdsio_type_2: "Storage->CPU->GPU (gdiso)",
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
        "downloads quantized GGUF models, ranging from a small 1.5B-parameter Qwen model to a "
        "standard 8B-parameter Llama 3 model, allowing performance testing across different VRAM "
        "and compute requirements."
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

    def __call__(
        self, benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...], docker_cleanup: bool
    ) -> Optional[BenchmarkExecutor]:
        """
        :param benchmark_name: To look up.
        :param gpus: List of GPUs to run the benchmark on. Jobs can utilize the GPU or decide not
        to use them.
        :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
        :return: Returns None if the wrapper module does not contain the desired benchmark.
        """
