"""
Set of types to describe benchmarking runs.
"""

from typing import Dict

from pydantic import BaseModel


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

    count: float
    result_max: float
    result_min: float
    mean: float
    std_dev: float


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
    cpu_count: int
    memory_mb: float
