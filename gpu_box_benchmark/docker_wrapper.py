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
"""
All containers receive a volume mounted temporary file so the benchmarks can optionally write
output directly to a file. This is the path the containers write to internally, the other side of
that volume mount. 
"""


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


class OutputsToResult(Protocol):
    """
    Defines functions that read container outputs and produce a numerical result.
    """

    def __call__(self, container_outputs: ContainerOutputs) -> float:
        """
        :param container_outputs: Container outputs, includes container logs and contents of the
        intermediate file which MAY OR MAY NOT have been used.
        :return: Numerical result as a float.
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


def _wait_get_logs(container: Container) -> Optional[str]:
    """
    Runs a `.wait()` on the input container and then pulls the stdout/stderr logs.
    Removes the container as well.
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


def _run_image_on_gpus(  # pylint: disable=too-many-positional-arguments
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

    # This removes the container.
    outputs = _wait_get_logs(container=container)

    return ContainerOutputs(
        logs=outputs,
        file=output_file_path,
    )


def _build_image(
    client: DockerClient, dockerfile_path: Path, tag: str, docker_cleanup: bool
) -> Image:
    """
    Canonical wrapper to build images.

    :param client: For interacting with docker.
    :param dockerfile_path: Path to the dockerfile to build. The build context will be the parent
    of this file.
    :param tag: Tag for the image.
    :param docker_cleanup: If True, efforts are made to leave little behind related to docker.
    :return: The built image.
    """

    LOGGER.debug("Building Image")

    if not docker_cleanup:
        try:
            LOGGER.debug(f"Checking for existing image {tag}...")
            return client.images.get(tag)
        except docker.errors.ImageNotFound:
            LOGGER.debug("Image not found locally. Proceeding to build.")

    build_directory = dockerfile_path.parent

    try:
        image, _build_logs = client.images.build(
            path=str(build_directory),
            dockerfile=str(dockerfile_path),
            tag=tag,
            nocache=docker_cleanup,
            rm=True,
            forcerm=True,
            pull=docker_cleanup,
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


def _force_parallel_run(  # pylint: disable=too-many-positional-arguments
    client: DockerClient,
    image: Image,
    new_output_file: Callable[[], Path],
    gpus: Tuple[GPUIdentity, ...],
    create_runtime_env_vars: CreateRuntimeEnvironmentVariables,
    outputs_to_result: OutputsToResult,
) -> List[float]:
    """
    Creates one docker container per GPU then starts each GPU at the same time. This is a simulation
    of the workload being multi-gpu capable but obviously there's no memory sharing on the GPUs
    happening.

    :param client: For interacting with docker.
    :param image: To run.
    :param new_output_file: Creates output files for the containers.
    :param gpus: GPUs to run on.
    :param create_runtime_env_vars: Creates environment variables if required.
    :param outputs_to_result: Converts outputs to a numerical result.
    :return: Results per GPU.
    """

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
        outputs_to_result(ContainerOutputs(logs=_wait_get_logs(container=container), file=file))
        for (container, file) in container_file
    ]

    return parallel_results


def _list_prunable_images(client: docker.DockerClient) -> List[Image]:
    """
    Lists images that would be removed kind of like `docker prune.
    :param client: For interacting with docker.
    :return: Image references.
    """

    # Image IDs referenced by any container (running or stopped)
    used_image_ids = {container.image.id for container in client.containers.list(all=True)}

    # Any image not referenced by a container is prunable
    return [image for image in client.images.list() if image.id not in used_image_ids]


def _docker_image_cleanup(client: DockerClient, run_image: Image, keep_images: List[Image]) -> None:
    """
    Removes dangling images and containers created throughout the benchmarking process.
    :param client: For interacting with Docker.
    :param run_image: Image that was run.
    :param keep_images: List of images to not touch.
    :return: None
    """

    client.images.remove(run_image.id, force=True)

    parent_id = run_image.attrs.get("Parent") if run_image else None

    if parent_id:
        try:
            # Check if it survived the prune before trying to remove
            client.images.get(parent_id)
            client.images.remove(parent_id)
            LOGGER.warning(f"Removed parent image: {parent_id}")
        except (docker.errors.ImageNotFound, docker.errors.APIError):
            pass

    keep_ids = {img.id for img in keep_images}
    used_image_ids = {container.image.id for container in client.containers.list(all=True)}

    cruft_images = list(
        filter(
            lambda image: (image.id not in keep_ids and image.id not in used_image_ids),
            client.images.list(),
        )
    )

    for cruft_image in cruft_images:
        # Safe to remove
        client.images.remove(cruft_image.id, force=True)
        LOGGER.debug(f"Removed cruft image: {cruft_image.id}")


def benchmark_dockerfile(  # pylint: disable=too-many-positional-arguments,too-many-locals
    dockerfile_path: Path,
    benchmark_name: str,
    benchmark_version: str,
    gpus: Tuple[GPUIdentity, ...],
    create_runtime_env_vars: CreateRuntimeEnvironmentVariables,
    outputs_to_result: OutputsToResult,
    multi_gpu_native: bool,
    docker_cleanup: bool,
) -> ReportFileNumerical:
    """
    Benchmarks a given dockerfile. The run sequence is as follows:

    0. The image is built and environment prepped.
    1. A container is run sequentially on _one_ of each type of GPUs passed in. So if there are four
    M10 cores and a P100, it is run on one of the M10 cores and the P100.
    2. A container is run on all the input GPUs in parallel.

    This strategy helps us understand the cost of parallelizing.

    Because all of these images and containers and images can become massive on disk, steps have
    been taken to prune and remove artifacts between tests. This makes the overall runtime slower.
    This shouldn't impact GPU scores and will likely be improved in the future.

    :param dockerfile_path: Path to the dockerfile to run.
    :param benchmark_name: Passed to image tagging process.
    :param benchmark_version: Passed to image tagging process.
    :param gpus: GPUs to run on.
    :param create_runtime_env_vars: Callable to create runtime environment variables.
    :param outputs_to_result: Converts the outputs from the container (logs and optional output
    file) to a numerical result.
    :param multi_gpu_native: If True, the input dockerfile can take advantage of having multiple
    GPUs passed in and utilize the multiple inputs simultaneously. If False, docker is used to
    run the benchmark on multiple GPUs at once.
    :param docker_cleanup: If given, run the docker image cleanup step after benchmarking.
    :return: Numerical results.
    """

    if not gpus:
        raise ValueError("Non GPU test aren't supported yet.")

    client = docker.from_env()

    # Images that would be pruned but existed on the system before the run. We don't want to
    # touch these because they might be used for something outside the scope of our app.
    existed_before_run = _list_prunable_images(client=client)

    image: Image = _build_image(
        client=client,
        dockerfile_path=dockerfile_path,
        tag="_".join([benchmark_name, benchmark_version]),
        docker_cleanup=docker_cleanup,
    )

    # If there's only a single GPU attached we can skip a lot of work.
    multiple_gpus_to_test = len(gpus) > 1

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

        first_result: float = next(iter(name_to_serial_result.values()))

        # Creates a docker container per GPU for every input GPU, then
        # starts them at the same time. Results are returned.
        parallel_results = (
            _force_parallel_run(
                client=client,
                image=image,
                new_output_file=new_output_file,
                gpus=gpus,
                create_runtime_env_vars=create_runtime_env_vars,
                outputs_to_result=outputs_to_result,
            )
            if multiple_gpus_to_test
            else [first_result]
        )

        native_multi_gpu_result: Optional[float] = None

        if multi_gpu_native:

            if multiple_gpus_to_test:

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

                native_multi_gpu_result = outputs_to_result(
                    ContainerOutputs(logs=complete_logs, file=multi_gpu_file)
                )
            else:
                native_multi_gpu_result = first_result

        theoretical_multi_gpu_sum = sum((name_to_serial_result[gpu.name] for gpu in gpus))
        forced_multi_gpu_sum = sum(parallel_results)

        if docker_cleanup:
            _docker_image_cleanup(
                client=client,
                run_image=image,
                keep_images=existed_before_run,
            )

        return ReportFileNumerical(
            min_by_gpu_type=min(name_to_serial_result.values()),
            max_by_gpu_type=max(name_to_serial_result.values()),
            mean_by_gpu_type=mean(name_to_serial_result.values()),
            theoretical_multi_gpu_mean=theoretical_multi_gpu_sum / len(gpus),
            theoretical_multi_gpu_sum=theoretical_multi_gpu_sum,
            forced_multi_gpu_numerical_mean=forced_multi_gpu_sum / len(gpus),
            forced_multi_gpu_sum=forced_multi_gpu_sum,
            native_multi_gpu_result=native_multi_gpu_result,
        )
