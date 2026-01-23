"""
CLI specific validation and common code.
"""

import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Optional, Tuple

from gpu_box_benchmark.locate_describe_hardware import (
    CPUIdentity,
    GPUIdentity,
    discover_cpu,
    discover_gpus,
)

BENCHMARK_COMMAND_NAME = "benchmark"

ENV_VAR_MAPPING = {
    "output_parent": "GBB_OUTPUT_PARENT",
    "title": "GBB_TITLE",
    "description": "GBB_DESCRIPTION",
}


def _path_safe(s: str, replacement: str = "_") -> str:
    """
    Strip out content that can cause problems in paths.
    :param s: To clean.
    :param replacement: Replacement seperator.
    :return: Clearn string.
    """

    # Normalize unicode (é → e, etc.)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    # Remove literal "(r)" (case-insensitive, optional)
    s = re.sub(r"\(r\)", "", s, flags=re.IGNORECASE)

    # Replace path separators and illegal characters (now including parentheses)
    s = re.sub(r'[<>:"/\\|?*()\x00-\x1F]', replacement, s)

    # Collapse whitespace and replacements
    s = re.sub(rf"{re.escape(replacement)}+", replacement, s)
    s = re.sub(r"\s+", replacement, s)

    # Strip leading/trailing separators
    return s.strip(replacement).strip()


def create_default_output_name(
    gpus: Tuple[GPUIdentity, ...] = (), cpu: Optional[CPUIdentity] = None, suffix: str = ".json"
) -> str:
    """
    CLI helper.
    :return: A name that encodes the current CPU and GPUs.
    """

    if not gpus:
        gpus = discover_gpus()

    cpu = discover_cpu() if cpu is None else cpu

    cpu_part = f"{_path_safe(str(cpu.physical_cpus))}_{_path_safe(cpu.name)}"

    name_to_counts = Counter([gpu.name for gpu in gpus])

    gpus_part = "_".join([f"{count}_{_path_safe(name)}" for name, count in name_to_counts.items()])

    date_part = _path_safe(datetime.now().replace(microsecond=0).isoformat())

    output = f"{cpu_part}_{gpus_part}_{date_part}{suffix}"

    return output
