"""
Code to find GPUs attached to the system and describe them for use in application code. Wraps
existing vendor-specific libraries.
"""

from typing import Tuple

import click
import nvsmi
from click import Context, Parameter
from pydantic import BaseModel


class GPUIdentity(BaseModel):
    """
    Describes a GPU. This struct is versioned along with the main version in SystemEvaluation.
    """

    # These fields are properties of the GPU itself, and how it is installed in the system.
    id: int
    name: str

    uuid: str
    """
    On NVIDIA GPUs with multiple cores per physical GPU, the UUID will be different for each
    of the cores on the same board. 
    """

    total_memory_mib: float
    driver: str

    serial: str
    """
    On NVIDIA GPUs with multiple cores per physical GPU, the serial will be the same for all cores 
    on the same GPU.
    """


def discover_gpus() -> Tuple[GPUIdentity, ...]:
    """
    Discover all GPUs attached to the system, returning them for use in the standard format.
    :return: A list of fully formed GPU Identities representing all the GPUs visible to the
    system.
    """

    nvidia_smi_gpus = nvsmi.get_gpus()

    output = tuple(
        map(
            lambda gpu: GPUIdentity(
                id=int(gpu.id),  # comes out of nvsmi as a string! Need to convert.
                name=gpu.name,
                uuid=gpu.uuid,
                total_memory_mib=gpu.mem_total,
                driver=gpu.driver,
                serial=gpu.serial,
            ),
            nvidia_smi_gpus,
        )
    )

    return output


def _gpu_option_callback(
    _ctx: Context,
    _param: Parameter,
    value: Tuple[int, ...],
) -> Tuple[GPUIdentity, ...]:
    """
    Handles conversion from the human-selectable GPU index to the format needed by the application.
    :param _ctx: Unused.
    :param _param: Unused
    :param value: List of GPU IDs to convert.
    :return: Converted IDs.
    """

    gpu_identities: Tuple[GPUIdentity, ...] = discover_gpus()

    return tuple(
        next((gpu_identity for gpu_identity in gpu_identities if gpu_identity.id == gpu_id))
        for gpu_id in value
    )


GPU_CLICK_OPTION = click.option(
    "--gpu",
    "-g",
    type=click.Choice(choices=[gpu_identity.id for gpu_identity in discover_gpus()]),
    help=(
        "The GPU(s) to use for computation. Can be given multiple times. "
        "If not given, all GPUs will be used."
    ),
    multiple=True,
    callback=_gpu_option_callback,
)
