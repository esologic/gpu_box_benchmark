"""
Utilities for finding GPUs attached to the system.

Need to eventually promote this to the other repos.
"""

import logging
from typing import Tuple, TypeAlias

import nvsmi

GPUDescription: TypeAlias = int

LOGGER = logging.getLogger(__name__)


def discover_gpus() -> Tuple[GPUDescription, ...]:
    """
    Returns the currently visible GPUs.
    :return: A tuple of GPUs.
    """

    gpus = tuple(int(gpu.id) for gpu in nvsmi.get_gpus())
    LOGGER.debug(f"Discovered {len(gpus)} GPUs: {gpus}")
    return gpus
