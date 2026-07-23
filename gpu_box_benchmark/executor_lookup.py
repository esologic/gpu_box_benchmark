"""
QOL code to make it easy create benchmark executors.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from gpu_box_benchmark.benchmark_dockerfile_wrappers import (
    ai_benchmark,
    blender_benchmark,
    content_aware_timelapse,
    folding_at_home,
    hashcat,
    llama_bench,
    nvidia_deep_learning_examples,
    nvidia_gds,
    whisper,
)
from gpu_box_benchmark.benchmark_types import (
    BenchmarkExecutor,
    BenchmarkName,
    CreateBenchmarkExecutor,
    NamedExecutor,
)
from gpu_box_benchmark.locate_describe_hardware import GPUIdentity


def render_named_executors(
    benchmarks: Tuple[BenchmarkName, ...],
    gpu: Tuple[GPUIdentity, ...],
    docker_cleanup: bool,
    force_rebuild: bool,
    downloads_directory_path: Optional[Path],
) -> List[Optional[NamedExecutor]]:
    """
    Create the set of functions that will run the benchmarks.

    :param benchmarks: Input from user, the requested tests.
    :param gpu: The GPUs to run the tests on.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :param force_rebuild: If given, the underlying docker containers are rebuilt from scratch
    before running. This will take more time that using the existing images.
    :param downloads_directory_path: Benchmark docker containers may need to download test assets
    during build time or runtime. By default, if this is not given, those assets are downloaded
    into the image. If this is given, the assets are downloaded to this location which is bind
    mounted into the container so the downloads can be re-used.
    :return: A list of executors for the input benchmarks. Will match input order.
    """

    creation_functions: List[CreateBenchmarkExecutor] = [
        nvidia_deep_learning_examples.create_resnet50_executor,
        llama_bench.create_llama_bench_throughput_executor,
        llama_bench.create_llama_bench_max_model_parameters_executor,
        blender_benchmark.create_blender_benchmark_executor,
        folding_at_home.create_fah_bench_executor,
        ai_benchmark.create_ai_benchmark_executor,
        whisper.create_whisper_executor,
        content_aware_timelapse.create_content_aware_timelapse_executor,
        nvidia_gds.create_gdsio_executor,
        hashcat.create_hashcat_executor,
    ]

    def executor_for_test(requested_test: BenchmarkName) -> BenchmarkExecutor:
        """
        Look up the executor for the input test. Adds logging.
        :param requested_test: From CLI.
        :return: Executor for the test.
        """
        try:
            return next(
                filter(
                    None,
                    (
                        creation_function(
                            benchmark_name=requested_test,
                            gpus=gpu,
                            docker_cleanup=docker_cleanup,
                            force_rebuild=force_rebuild,
                            downloads_directory_path=downloads_directory_path,
                        )
                        for creation_function in creation_functions
                    ),
                ),
            )
        except StopIteration as lookup_error:
            raise StopIteration(
                f"Couldn't find executor for test: {requested_test.value}"
            ) from lookup_error

    try:
        return [
            NamedExecutor(
                benchmark_name=b_m,
                executor=executor_for_test(requested_test=b_m),
            )
            for b_m in benchmarks
        ]
    except StopIteration as e:
        raise ValueError("No valid tests for the given input.") from e
