"""
UT suite to really run tests given the currently attached hardware.
"""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Tuple

import pytest

from gpu_box_benchmark.benchmark_types import BenchmarkName
from gpu_box_benchmark.benchmark_types import BenchmarkResult
from gpu_box_benchmark.executor_lookup import render_named_executors
from gpu_box_benchmark.locate_describe_hardware import discover_gpus


_NONZERO_NUMERICAL_RESULT_KEYS = (
    "min_by_gpu_type",
    "max_by_gpu_type",
    "mean_by_gpu_type",
    "theoretical_multi_gpu_mean",
    "theoretical_multi_gpu_sum",
    "forced_multi_gpu_numerical_mean",
    "forced_multi_gpu_sum",
)


@contextmanager
def assets_directory() -> Iterator[Path]:
    """
    Yield the NAS models directory if available, otherwise a temporary directory.
    :return: Context manager with the directory.
    """

    nas_path = Path("/mnt/network_storage/Devon Bray/Active Projects/models")

    if nas_path.exists():
        yield nas_path
        return

    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.mark.parametrize(
    "benchmarks",
    [
        (BenchmarkName.blender_benchmark_monster_gpu,),
        (BenchmarkName.llama_bench_meta_llama_3_8b_instruct_generation,),
        (BenchmarkName.hashcat_sha256,),
        (BenchmarkName.llama_bench_max_model_parameters_q4_32k_ctx,),
        (BenchmarkName.llama_bench_qwen_2_5_1_5b_instruct_prompt,),
    ],
)
def test_run_benchmarks(benchmarks: Tuple[BenchmarkName, ...]) -> None:
    """
    Actually runs benchmarks with the currently attached GPUs to make sure the happy path works.
    :param benchmarks: Sets of benchmarks to run.
    :return: None
    """

    with assets_directory() as downloads_directory_path:
        executors = render_named_executors(
            benchmarks=benchmarks,
            gpu=discover_gpus(),
            docker_cleanup=False,
            force_rebuild=False,
            downloads_directory_path=downloads_directory_path,
        )

        for named_executor in executors:
            if named_executor is not None:
                result: BenchmarkResult = named_executor.executor()

                # Happy path is all tests produce numerical values in this set of keys.
                numerical_results_dict = result.numerical_results.model_dump()

                assert all(numerical_results_dict[key] for key in _NONZERO_NUMERICAL_RESULT_KEYS)
            else:
                pytest.fail("No executor found.")
