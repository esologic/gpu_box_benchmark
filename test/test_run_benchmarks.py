"""
UT suite to really run tests given the currently attached hardware.
"""

import pytest
from typing import Tuple
from gpu_box_benchmark.benchmark_types import BenchmarkName
from gpu_box_benchmark.benchmark_types import BenchmarkResult
from gpu_box_benchmark.locate_describe_hardware import (
    discover_gpus,
)

from gpu_box_benchmark.executor_lookup import render_named_executors


@pytest.mark.parametrize(
    "benchmarks",
    [
        (BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_generation,),
    ],
)
def test_run_benchmarks(benchmarks: Tuple[BenchmarkName, ...]) -> None:
    """
    Actually runs benchmarks with the currently attached GPUs to make sure the happy path works.
    :param benchmarks: Sets of benchmarks to run.
    :return: None
    """

    executors = render_named_executors(
        benchmarks=benchmarks,
        gpu=discover_gpus(),
        docker_cleanup=True,
    )

    for named_executor in executors:

        if named_executor is not None:
            _result: BenchmarkResult = named_executor.executor()
        else:
            pytest.fail("No executor found.")
