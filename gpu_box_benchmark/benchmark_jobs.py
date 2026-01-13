"""
Abstractions to define different named benchmarks and how they get run.
"""

from enum import Enum
from typing import Optional, Protocol, Tuple

from gpu_box_benchmark.gpu_discovery import GPUDescription
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult


class BenchmarkName(str, Enum):
    """
    The different benchmarks the user can choose from.
    """

    resnet50_train_batch_1_amp = "resnet50_train_batch_1_amp"
    resnet50_train_batch_64_amp = "resnet50_train_batch_64_amp"
    resnet50_infer_batch_1_amp = "resnet50_infer_batch_1_amp"
    resnet50_infer_batch_256_amp = "resnet50_infer_batch_256_amp"


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
        self, benchmark_name: BenchmarkName, gpus: Tuple[GPUDescription, ...]
    ) -> Optional[BenchmarkExecutor]:
        """
        :param benchmark_name: To look up.
        :param gpus: List of GPUs to run the benchmark on. Jobs can utilize the GPU or decide not
        to use them.
        :return: Returns None if the wrapper module does not contain the desired benchmark.
        """
