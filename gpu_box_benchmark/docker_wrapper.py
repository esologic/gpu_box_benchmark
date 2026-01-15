import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import docker
from docker.errors import APIError, BuildError

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity

LOGGER = logging.getLogger(__name__)


def build_run_dockerfile_read_logs(
    dockerfile_path: Path,
    tag: str,
    gpus: Tuple[GPUIdentity, ...],
    env_vars: List[Tuple[str, Union[str, float, bool, int]]],
) -> Optional[str]:
    """
    Builds and runs a dockerfile, returning the logs from the container.
    :param dockerfile_path: Path to the dockerfile. Parent to this path will be used as the build
    directory.
    :param tag: Image will be assigned this tag.
    :param gpus: Docker container will have access to these GPUs.
    :param env_vars: List of Tuples of string and whatever that will be passed as environment
    variables to the container at runtime.
    :return: Logs from the container.
    """

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

    LOGGER.debug("Image Built. Running")

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
