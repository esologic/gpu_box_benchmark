"""
Abstractions to define different named benchmarks and how they get run.
"""

from enum import Enum
from typing import Optional, Protocol, Tuple

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult


class BenchmarkName(str, Enum):
    """
    The different benchmarks the user can choose from.
    """

    resnet50_train_batch_1_amp = "resnet50_train_batch_1_amp"
    resnet50_train_batch_64_amp = "resnet50_train_batch_64_amp"
    resnet50_infer_batch_1_amp = "resnet50_infer_batch_1_amp"
    resnet50_infer_batch_256_amp = "resnet50_infer_batch_256_amp"

    llama_bench_tiny_model_prompt = "llama_bench_tiny_model_prompt"
    llama_bench_tiny_model_generation = "llama_bench_tiny_model_generation"
    llama_bench_standard_model_prompt = "llama_bench_standard_model_prompt"
    llama_bench_standard_model_generation = "llama_bench_standard_model_generation"
    """
    Prompt processing is compute speed bound. Very paralellizable. 
    Token generation is memory bandwidth bound. This is a good test to see how fast data can move
    from card to card and is a good test of the PCIe architecture. 
    """

    blender_monster_cpu = "blender_monster_cpu"
    blender_monster_gpu = "blender_monster_gpu"

    fah_bench_single = "fah_bench_single"
    fah_bench_double = "fah_bench_double"


class BenchmarkExecutor(Protocol):
    """
    Defines the callables that run to produce the benchmark results.
    """

    def __call__(self) -> NumericalBenchmarkResult:
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
        self, benchmark_name: BenchmarkName, gpus: Tuple[GPUIdentity, ...]
    ) -> Optional[BenchmarkExecutor]:
        """
        :param benchmark_name: To look up.
        :param gpus: List of GPUs to run the benchmark on. Jobs can utilize the GPU or decide not
        to use them.
        :return: Returns None if the wrapper module does not contain the desired benchmark.
        """
