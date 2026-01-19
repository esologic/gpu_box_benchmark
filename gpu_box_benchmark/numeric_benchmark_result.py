"""
Set of types to describe benchmarking runs.
"""

from typing import Dict, List, Tuple

from pydantic import BaseModel

from gpu_box_benchmark.locate_describe_hardware import CPUIdentity, GPUIdentity


class ReportFileNumerical(BaseModel):
    """
    Contain the numerical results for the run
    """

    min_by_gpu_type: float
    max_by_gpu_type: float
    mean_by_gpu_type: float

    theoretical: float
    experimental: float


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

    cpu: CPUIdentity

    total_memory_gb: float

    gpus: Tuple[GPUIdentity, ...]

    results: List[BenchmarkResult]
