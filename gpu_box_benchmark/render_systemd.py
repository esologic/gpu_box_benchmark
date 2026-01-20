"""
Helps create systemd unit files.
"""

import os
import sys
from pathlib import Path

from jinja2 import Template

from gpu_box_benchmark import assets
from gpu_box_benchmark.cli_common import BENCHMARK_COMMAND_NAME, ENV_VAR_MAPPING


def render_systemd_file(
    path_to_python_file: str, output_parent: Path, title: str, description: str
) -> str:
    """
    Coerce the input args to strings and populate the asset systemd unit file.
    Way more options are provided in the standalone mode, but we want the systemd runs to use CLI
    defaults to automatically discover all GPUs and tests and run everything it can.
    :param path_to_python_file: Path to the python file to run.
    :param output_parent: Benchmark files are written into this directory.
    :param title: Title passed as env var.
    :param description: Description passed as env var.
    """

    def systemd_escape(value: str) -> str:
        """
        Escape a value for systemd Environment= line.

        - Backslashes are escaped first
        - Double quotes are escaped
        - Dollar signs are doubled
        """
        value = str(value)  # in case it’s not a string
        value = value.replace("\\", "\\\\")
        value = value.replace('"', '\\"')
        value = value.replace("$", "$$")
        return value

    template = Template(assets.SYSTEMD_TEMPLATE_PATH.read_text(encoding="utf-8"))

    rendered = template.render(
        user=os.getlogin(),
        exec_start=" ".join([sys.executable, path_to_python_file, BENCHMARK_COMMAND_NAME]),
        env_vars={
            k: systemd_escape(str(v))
            for k, v in {
                ENV_VAR_MAPPING["output_parent"]: output_parent.resolve(),
                ENV_VAR_MAPPING["title"]: title,
                ENV_VAR_MAPPING["description"]: description,
            }.items()
        },
    )

    return rendered
