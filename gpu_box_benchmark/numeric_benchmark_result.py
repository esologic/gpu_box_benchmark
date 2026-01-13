"""
Set of types to describe benchmarking runs.
"""

from typing import Dict, List, Tuple

from pydantic import BaseModel

from gpu_box_benchmark.locate_describe_gpu import GPUIdentity


class NumericalBenchmarkResult(BaseModel):
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

    verbose_unit: str
    unit: str

    sample_count: float

    percentile_25: float
    percentile_50: float
    percentile_75: float
    mean: float
    std: float
    result_min: float
    result_max: float


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

    cpu_name: str

    physical_cpus: int
    logical_cpus: int

    total_memory_gb: float

    gpus: Tuple[GPUIdentity, ...]

    results: List[NumericalBenchmarkResult]
