"""
Convince wrappers around the docker python API.
"""

import logging
import tempfile
import uuid
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple

import docker
from docker.client import DockerClient
from docker.errors import APIError, BuildError
from docker.models.containers import Container
from docker.models.images import Image

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity
from gpu_box_benchmark.numeric_benchmark_result import ReportFileNumerical

LOGGER = logging.getLogger(__name__)

_INTERNAL_CONTAINER_OUTPUT_FILE_PATH = "/results/result.txt"


class ContainerOutputs(NamedTuple):
    """
    Set of container outputs.
    """

    logs: str
    """
    Stdout and Stderr logs as a string.
    """

    file: Path
    """
    Each container is provided a path to write output as a file. Using it is up to the container.
    This is a reference to that file.
    """


class CreateRuntimeEnvironmentVariables(Protocol):
    """
    Defines the function to create runtime environment variables. Some containers need input
    about the input GPUs to be able to be configured correctly.
    """

    def __call__(self, runtime_gpus: Tuple[GPUIdentity, ...]) -> List[Tuple[str, str]]:
        """
        :param runtime_gpus: GPUs that will be passed to the container for this run.
        :return: Environment variables as a list of tuples.
        """


def _create_gpu_container(
    client: DockerClient,
    image: Image,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, str]],
    output_file_path: Path,
) -> Container:
    """
    Wrapper around the container creation function to pre-fill the params needed for GPU usage.

    :param image: Built image to run.
    :param gpus: GPUs to run the container on.
    :param env_vars: Passed as environment variables to the container. Values are converted to
    strings before passing.
    :param output_file_path: Some containers output via stdout, some containers need to write their
    results to a file. This path is volume mounted into the container so that it may optionally
    write to it.
    :return: The unstarted container.
    """

    container = client.containers.create(
        image=image,
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=list(str(gpu.id) for gpu in gpus),
                capabilities=[["gpu"]],
            )
        ],
        environment=dict(env_vars),
        ipc_mode="host",
        detach=True,
        volumes={
            str(output_file_path): {
                # This location is baked into all dockerfiles used by the benchmarking suite that
                # need to output to a file.
                "bind": _INTERNAL_CONTAINER_OUTPUT_FILE_PATH,
                "mode": "rw",
            }
        },
    )

    return container


def _wait_get_logs(container: Container) -> str:
    """
    Runs a `.wait()` on the input container and then pulls the stdout/stderr logs.
    :param container: To interact with.
    :return: The logs as a single string.
    """

    logs: Optional[str] = None

    try:

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

    if logs is None:
        raise ValueError("Couldn't read logs from container")

    return logs


def _run_image_on_gpus(
    client: DockerClient,
    image: Image,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, str]],
    output_file_path: Path,
) -> Optional[ContainerOutputs]:
    """
    Creates a container, starts it, and waits for its completion on a set of GPUs. This blocks
    on waiting for the container to exit.

    :param client: For interacting with docker.
    :param image: To run.
    :param gpus: GPUs to run the container on.
    :param env_vars: Passed as environment.
    :param output_file_path: Passed into container for optional usage.
    :return: The set of outputs from the container.
    """

    container = _create_gpu_container(
        client=client, image=image, gpus=gpus, env_vars=env_vars, output_file_path=output_file_path
    )

    container.start()

    return ContainerOutputs(logs=_wait_get_logs(container=container), file=output_file_path)


def _build_image(
    client: DockerClient,
    dockerfile_path: Path,
    tag: str,
) -> Image:
    """
    Canonical wrapper to build images.

    :param client: For interacting with docker.
    :param dockerfile_path: Path to the dockerfile to build. The build context will be the parent
    of this file.
    :param tag: Tag for the image.
    :return: The built image.
    """

    LOGGER.debug("Building Image")

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


def benchmark_dockerfile(  # pylint: disable=too-many-positional-arguments,too-many-locals
    dockerfile_path: Path,
    tag: str,
    gpus: Tuple[GPUIdentity, ...],
    create_runtime_env_vars: CreateRuntimeEnvironmentVariables,
    outputs_to_result: Callable[[ContainerOutputs], float],
    multi_gpu_native: bool,
) -> ReportFileNumerical:
    """
    Benchmarks a given dockerfile. The run sequence is as follows:

    0. The image is built and environment prepped.
    1. A container is run sequentially on _one_ of each type of GPUs passed in. So if there are four
    M10 cores and a P100, it is run on one of the M10 cores and the P100.
    2. A container is run on all the input GPUs in parallel.

    This strategy helps us understand the cost of parallelizing.

    :param dockerfile_path: Path to the dockerfile to run.
    :param tag: Tage for the image.
    :param gpus: GPUs to run on.
    :param create_runtime_env_vars: Callable to create runtime environment variables.
    :param outputs_to_result: Converts the outputs from the container (logs and optional output
    file) to a numerical result.
    :param multi_gpu_native: If True, the input dockerfile can take advantage of having multiple
    GPUs passed in and utilize the multiple inputs simultaneously. If False, docker is used to
    run the benchmark on multiple GPUs at once.
    :return: Numerical results.
    """

    client = docker.from_env()

    image = _build_image(client=client, dockerfile_path=dockerfile_path, tag=tag)

    # Select the lowest ID's GPU in each of the input GPUs grouped by GPU name.
    serial_gpus: List[GPUIdentity] = [
        next(iter(sorted([gpu for gpu in gpus if gpu.name == name], key=lambda gpu: gpu.id)))
        for name in {gpu.name for gpu in gpus}
    ]

    with tempfile.TemporaryDirectory() as td:

        temporary_directory = Path(td)

        def new_output_file() -> Path:
            """
            :return: Path to a new file in the temporary directory.
            """

            output_file = temporary_directory / str(uuid.uuid4().hex)
            output_file.touch()

            return output_file

        # A single datapoint per type of GPU attached to the system. When the benchmark is run on
        # all attached GPUs at the same time this lets us understand how much overhead is added.
        name_to_serial_result: Dict[str, float] = {
            gpu.name: outputs_to_result(
                _run_image_on_gpus(
                    client=client,
                    image=image,
                    gpus=(gpu,),
                    env_vars=create_runtime_env_vars(runtime_gpus=(gpu,)),
                    output_file_path=new_output_file(),
                )
            )
            for gpu in serial_gpus
        }

        if not multi_gpu_native:

            gpu_file: List[Tuple[GPUIdentity, Path]] = [(gpu, new_output_file()) for gpu in gpus]

            container_file: List[Tuple[Container, Path]] = [
                (
                    _create_gpu_container(
                        client=client,
                        image=image,
                        gpus=(gpu,),
                        env_vars=create_runtime_env_vars(runtime_gpus=(gpu,)),
                        output_file_path=file,
                    ),
                    file,
                )
                for (gpu, file) in gpu_file
            ]

            # Start all containers at around the same time
            for container, _ in container_file:
                container.start()

            parallel_results: List[float] = [
                outputs_to_result(
                    ContainerOutputs(logs=_wait_get_logs(container=container), file=file)
                )
                for (container, file) in container_file
            ]

            experimental: float = sum(parallel_results)
            theoretical_result = sum((name_to_serial_result[gpu.name] for gpu in gpus))

        else:

            multi_gpu_file = new_output_file()

            multi_gpu_container = _create_gpu_container(
                client=client,
                image=image,
                gpus=gpus,
                env_vars=create_runtime_env_vars(runtime_gpus=gpus),
                output_file_path=multi_gpu_file,
            )
            multi_gpu_container.start()

            complete_logs = _wait_get_logs(container=multi_gpu_container)

            experimental = outputs_to_result(
                ContainerOutputs(logs=complete_logs, file=multi_gpu_file)
            )
            theoretical_result = mean([name_to_serial_result[gpu.name] for gpu in gpus])

        return ReportFileNumerical(
            min_by_gpu_type=min(name_to_serial_result.values()),
            max_by_gpu_type=max(name_to_serial_result.values()),
            mean_by_gpu_type=mean(name_to_serial_result.values()),
            theoretical=theoretical_result,
            experimental=experimental,
        )
