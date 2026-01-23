"""
Tests the library functions for interacting with docker.
"""

import pytest
from gpu_box_benchmark import docker_wrapper
from test.assets import HELLO_WORLD_DOCKERFILE_PATH
from gpu_box_benchmark import locate_describe_hardware
from gpu_box_benchmark.numeric_benchmark_result import ReportFileNumerical


@pytest.mark.integration
def test_benchmark_dockerfile_small() -> None:
    """
    Runs a very simple and small dockerfile to make sure the pipeline works as expected.
    :return: None
    """

    result = docker_wrapper.benchmark_dockerfile(
        dockerfile_path=HELLO_WORLD_DOCKERFILE_PATH,
        tag_prefix="hello_world_test",
        gpus=locate_describe_hardware.discover_gpus(),
        create_runtime_env_vars=lambda runtime_gpus: [],
        outputs_to_result=lambda container_outputs: float(
            container_outputs.logs == "hello world\n"
        ),
        multi_gpu_native=False,
    )

    assert (
        ReportFileNumerical(
            min_by_gpu_type=1.0,
            max_by_gpu_type=1.0,
            mean_by_gpu_type=1.0,
            theoretical_multi_gpu_mean=1.0,
            theoretical_multi_gpu_sum=2.0,
            forced_multi_gpu_numerical_mean=1.0,
            forced_multi_gpu_sum=2.0,
            native_multi_gpu_result=None,
        )
        == result
    )
