"""
Set of types to describe benchmarking runs.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from gpu_box_benchmark.locate_describe_hardware import CPUIdentity, GPUIdentity


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
    Contain the numerical results for the run. Units etc are kept in the outer result.
    """

    min_by_gpu_type: float
    max_by_gpu_type: float
    mean_by_gpu_type: float

    theoretical_multi_gpu_mean: float
    theoretical_multi_gpu_sum: float

    forced_multi_gpu_numerical_mean: float
    forced_multi_gpu_sum: float

    native_multi_gpu_result: Optional[float]
    """
    This could either map onto the theoretical sum _or_ mean depending on the benchmark. If it 
    becomes hard to distinguish here, add a a flag to the outer type. None here means there's no
    native multi-gpu scoring supported.
    """


class BenchmarkResult(BaseModel):
    """
    Describes an individual benchmark result. This should be the output of a single test.
    """

    name: str
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
