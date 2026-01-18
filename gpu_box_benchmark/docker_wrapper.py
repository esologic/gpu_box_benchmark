"""
Convince wrappers around the docker python API.
"""

import logging
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import docker
from docker.errors import APIError, BuildError
from docker.models.containers import Container
from docker.models.images import Image

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import NumericalBenchmarkResult, ReportFileNumerical

LOGGER = logging.getLogger(__name__)


def create_gpu_container(
    image: Image,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, Union[str, float, bool, int]]],
) -> Container:

    client = docker.from_env()

    container = client.containers.create(
        image=image,
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=list(str(gpu.id) for gpu in gpus),
                capabilities=[["gpu"]],
            )
        ],
        environment={key: str(value) for key, value in env_vars},
        ipc_mode="host",
        detach=True,
    )

    return container


def run_image_on_gpus(
    image: Image,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, Union[str, float, bool, int]]],
) -> Optional[str]:

    container = create_gpu_container(image=image, gpus=gpus, env_vars=env_vars)

    logs: Optional[str] = None

    try:
        container.start()
        result = container.wait()  # blocks until exit

        logs = container.logs(stdout=True, stderr=True).decode()
        status_code = result["StatusCode"]

        if status_code != 0:
            LOGGER.error("Container failed with exit code %s", status_code)
            LOGGER.error("Container logs:\n%s", logs)
            raise RuntimeError("Container execution failed")

        LOGGER.debug("Container completed successfully")
        LOGGER.debug("Container logs:\n%s", logs)

    finally:
        container.remove(force=True)

    return logs


def build_image(
    dockerfile_path: Path,
    tag: str,
) -> Image:

    LOGGER.debug("Building Image")

    client = docker.from_env()

    build_directory = dockerfile_path.parent

    try:
        image, _build_logs = client.images.build(
            path=str(build_directory),
            dockerfile=str(dockerfile_path),
            nocache=False,
            tag=tag,
        )

        return image

    except BuildError as e:
        LOGGER.error("Docker build failed!\n")

        for entry in e.build_log:
            # entries are dicts with keys like 'stream', 'error', 'status'
            if "stream" in entry:
                LOGGER.error(entry["stream"])
            elif "error" in entry:
                LOGGER.error(entry["error"])

        raise  # re-raise if you want the failure to propagate

    except APIError:
        # Docker daemon / API-level failure
        LOGGER.error("Docker API error", exc_info=True)
        raise


def wait_get_logs(container: Container) -> str:

    result = container.wait()  # blocks until exit

    logs = container.logs(stdout=True, stderr=True).decode()
    status_code = result["StatusCode"]

    if status_code != 0:
        LOGGER.error("Container failed with exit code %s", status_code)
        LOGGER.error("Container logs:\n%s", logs)
        raise RuntimeError("Container execution failed")

    LOGGER.debug("Container completed successfully")
    LOGGER.debug("Container logs:\n%s", logs)

    return logs


def build_run_dockerfile_read_logs(
    dockerfile_path: Path,
    tag: str,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, Union[str, float, bool, int]]],
) -> Optional[str]:
    """
    Builds and runs a dockerfile, returning the logs from the container.
    Removes the container as well.
    :param dockerfile_path: Path to the dockerfile. Parent to this path will be used as the build
    directory.
    :param tag: Image will be assigned this tag.
    :param gpus: Docker container will have access to these GPUs.
    :param env_vars: List of Tuples of string and whatever that will be passed as environment
    variables to the container at runtime.
    :return: Logs from the container.
    """

    client = docker.from_env()

    image = build_image(dockerfile_path=dockerfile_path, tag=tag)

    LOGGER.debug("Image Built. Running")

    container = create_gpu_container(image=image, gpus=gpus, env_vars=env_vars)

    logs: Optional[str] = None

    try:
        container.start()
        logs = wait_get_logs(container)

    finally:
        container.remove(force=True)

    return logs


def benchmark_dockerfile(
    dockerfile_path: Path,
    tag: str,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, Union[str, float, bool, int]]],
    logs_to_result: Callable[[str], float],
    multi_gpu_native: bool,
) -> ReportFileNumerical:
    """

    :param dockerfile_path:
    :param tag:
    :param gpus:
    :param env_vars:
    :return:
    """

    image = build_image(dockerfile_path=dockerfile_path, tag=tag)

    # Select the lowest ID's GPU in each of the input GPUs grouped by GPU name.
    serial_gpus: List[GPUIdentity] = [
        next(iter(sorted([gpu for gpu in gpus if gpu.name == name], key=lambda gpu: gpu.id)))
        for name in {gpu.name for gpu in gpus}
    ]

    # A single datapoint per type of GPU attached to the system. When the benchmark is run on
    # all attached GPUs at the same time this lets us understand how much overhead is added.
    name_to_serial_result: Dict[str, float] = {
        gpu.name: logs_to_result(run_image_on_gpus(image=image, gpus=(gpu,), env_vars=env_vars))
        for gpu in serial_gpus
    }

    if not multi_gpu_native:
        container_per_gpu: List[Container] = [
            create_gpu_container(image=image, gpus=(gpu,), env_vars=env_vars) for gpu in gpus
        ]

        # Start all containers at around the same time
        for container in container_per_gpu:
            container.start()

        parallel_results = [
            logs_to_result(wait_get_logs(container=container)) for container in container_per_gpu
        ]

        parallel_mean = mean(parallel_results)
        experimental_sum = sum(parallel_results)

    else:
        container = create_gpu_container(image=image, gpus=gpus, env_vars=env_vars)

        experimental_sum = logs_to_result(wait_get_logs(container=container))
        parallel_mean = experimental_sum / len(gpus)

    return ReportFileNumerical(
        min_by_gpu_type=min(name_to_serial_result.values()),
        max_by_gpu_type=max(name_to_serial_result.values()),
        mean_by_gpu_type=mean(name_to_serial_result.values()),
        theoretical_sum=sum([name_to_serial_result[gpu.name] for gpu in gpus]),
        parallel_mean=parallel_mean,
        experimental_sum=experimental_sum,
    )
