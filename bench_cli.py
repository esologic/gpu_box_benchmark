"""Main module."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import click
import docker

from gpu_box_benchmark import nvidia_deep_learning_examples_wrapper

_DIRECTORY_ROOT = Path(__file__).parent.resolve()

BENCHMARK_DOCKERFILE_DIR = _DIRECTORY_ROOT / "benchmark_dockerfiles"

LOGGER_FORMAT = "[%(asctime)s - %(process)s - %(name)20s - %(levelname)s] %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)

LOGGER = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """
    Uses Docker to run GPU related benchmarks with the goal of comparing system hardware
    architectures.

    \f

    :return: None
    """


@cli.command(short_help="Run one or more benchmarks and records the results.")
def benchmark() -> (  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    None
):
    """
    Run one or more benchmarks and records the results.

    \f

    TODO: We may want to check how loaded down the system is before starting benchmarks.

    :return: None
    """

    client = docker.from_env()

    LOGGER.info("Building Image")

    build_directory = BENCHMARK_DOCKERFILE_DIR / "resnet50"

    image, _logs = client.images.build(
        path=str(build_directory),
        dockerfile=str(build_directory / "Dockerfile"),
        nocache=False,
        tag="resnet50",
    )

    LOGGER.info("Image Built. Running")

    mode_training = True

    with NamedTemporaryFile(suffix=".txt") as f:

        client.containers.run(
            image=image,
            auto_remove=True,
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=["0"],
                    capabilities=[["gpu"]],
                )
            ],
            environment={
                "BATCH_SIZE": "32",
                "NGPUS": "1",
                "MODE_TRAINING": str(int(mode_training)),
            },
            volumes={
                str(f.name): {
                    "bind": "/results/output.txt",
                    "mode": "rw",
                }
            },
        )

        f.seek(0)

        print(
            nvidia_deep_learning_examples_wrapper.parse_report_file(
                Path(f.name), mode_training=mode_training
            )
        )


if __name__ == "__main__":
    cli()
