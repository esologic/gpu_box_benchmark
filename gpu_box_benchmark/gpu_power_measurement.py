"""
Measure GPU energy consumption using NVIDIA NVML.

Energy is estimated by periodically sampling instantaneous GPU power and integrating
power over the elapsed time.
"""

import csv
import queue
import subprocess
import threading
import time
from contextlib import AbstractContextManager
from typing import Dict, Tuple

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity

_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1


def _read_gpu_power_watts(
    gpus: Tuple[GPUIdentity, ...],
) -> Dict[str, float]:
    """
    Read current GPU power usage for the given GPUs.

    :param gpus: GPUs to query.
    :return: Current power usage in watts keyed by GPU UUID.
    """
    run_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,power.draw",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )

    results = [
        dict(zip(("index", "power.draw"), line))
        for line in csv.reader(
            run_output.stdout.decode().splitlines(),
            skipinitialspace=True,
        )
    ]

    result_by_index = {int(result["index"]): float(result["power.draw"]) for result in results}

    return {gpu.uuid: result_by_index[gpu.id] for gpu in gpus}


def _measure_gpu_energy(
    gpus: Tuple[GPUIdentity, ...],
    stop_event: threading.Event,
    result_queue: queue.Queue[Dict[str, float]],
    sample_interval_seconds: float,
) -> None:
    """
    Measure GPU energy consumption until stop_event is set.

    All measurement state is local to this thread. The final measurements are
    passed back through result_queue.
    """
    energy_joules = {gpu.uuid: 0.0 for gpu in gpus}
    previous_time = time.monotonic()

    while True:
        power_watts_by_gpu = _read_gpu_power_watts(gpus)

        current_time = time.monotonic()
        elapsed_seconds = current_time - previous_time
        previous_time = current_time

        for uuid, power_watts in power_watts_by_gpu.items():
            energy_joules[uuid] += power_watts * elapsed_seconds

        if stop_event.wait(sample_interval_seconds):
            break

    result_queue.put(energy_joules)


class GPUPowerMeasurement(AbstractContextManager["GPUPowerMeasurement"]):
    """
    Measure GPU energy consumption for the duration of a context.

    Usage:

        with GPUPowerMeasurement(gpus) as measurement:
            run_benchmark()

        energy_joules = measurement.energy_joules
    """

    def __init__(
        self,
        gpus: Tuple[GPUIdentity, ...],
        sample_interval_seconds: float = _DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._gpus = gpus
        self._sample_interval_seconds = sample_interval_seconds

        self._stop_event = threading.Event()
        self._result_queue: queue.Queue[Dict[str, float]] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

        self._energy_joules: Dict[str, float] | None = None

    def __enter__(self) -> "GPUPowerMeasurement":
        if self._thread is not None:
            raise RuntimeError("GPU power measurement has already been started")

        self._thread = threading.Thread(
            target=_measure_gpu_energy,
            args=(
                self._gpus,
                self._stop_event,
                self._result_queue,
                self._sample_interval_seconds,
            ),
            name="gpu-power-measurement",
            daemon=True,
        )
        self._thread.start()

        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._thread is None:
            raise RuntimeError("GPU power measurement was never started")

        self._stop_event.set()
        self._thread.join()

        self._energy_joules = self._result_queue.get()

    @property
    def energy_joules(self) -> Dict[str, float]:
        """
        Return the measured energy consumption for each GPU.

        This property is only valid after the context manager has exited.
        """
        if self._energy_joules is None:
            raise RuntimeError(
                "GPU power measurement results are not available until the "
                "measurement context has exited"
            )

        return self._energy_joules
