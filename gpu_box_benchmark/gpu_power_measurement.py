"""
Measure GPU energy consumption using NVIDIA ``nvidia-smi``.

Energy is estimated by periodically sampling instantaneous GPU power and integrating
power over the elapsed time.
"""

import csv
import logging
import queue
import subprocess
import threading
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

from gpu_box_benchmark.locate_describe_hardware import GPUIdentity

_QUEUE_INTERACTION_TIMEOUT = 0.5

_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1
_NVIDIA_SMI_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"

_MeasurementResult = Union[Dict[str, float], Exception]

LOGGER = logging.getLogger(__name__)


def _read_gpu_power_watts(
    gpus: Tuple[GPUIdentity, ...],
) -> Dict[str, Tuple[datetime, float]]:
    """
    Read GPU power usage and measurement timestamps.

    :param gpus: GPUs to query.
    :return: Timestamp and power in watts keyed by GPU UUID.
    """

    run_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,power.draw",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )

    results = [
        dict(zip(("timestamp", "index", "power.draw"), line))
        for line in csv.reader(
            run_output.stdout.decode().splitlines(),
            skipinitialspace=True,
        )
    ]

    result_by_index = {
        int(result["index"]): (
            datetime.strptime(result["timestamp"], _NVIDIA_SMI_TIMESTAMP_FORMAT),
            float(result["power.draw"]),
        )
        for result in results
    }

    return {gpu.uuid: result_by_index[gpu.id] for gpu in gpus}


def _measure_gpu_energy(
    gpus: Tuple[GPUIdentity, ...],
    stop_event: threading.Event,
    result_queue: queue.Queue[_MeasurementResult],
    sample_interval_seconds: float,
) -> None:
    """
    Measure GPU energy until the stop event is set.

    :param gpus: GPUs to measure.
    :param stop_event: Event used to stop measurement.
    :param result_queue: Queue used to return the result.
    :param sample_interval_seconds: Minimum delay between measurements.
    """

    try:
        energy_joules: Dict[str, float] = {gpu.uuid: 0.0 for gpu in gpus}

        previous_measurements = _read_gpu_power_watts(gpus)

        while True:

            measurements = _read_gpu_power_watts(gpus)

            for gpu in gpus:

                previous_timestamp, previous_power_watts = previous_measurements[gpu.uuid]
                timestamp, power_watts = measurements[gpu.uuid]

                elapsed_seconds = (timestamp - previous_timestamp).total_seconds()

                energy_increment_joules = power_watts * elapsed_seconds
                energy_joules[gpu.uuid] += energy_increment_joules

                LOGGER.debug(
                    f"GPU power measurement: "
                    f"uuid={gpu.uuid} "
                    f"timestamp={timestamp.isoformat()} "
                    f"power_watts={power_watts:.3f} "
                    f"previous_timestamp={previous_timestamp.isoformat()} "
                    f"previous_power_watts={previous_power_watts:.3f} "
                    f"elapsed_seconds={elapsed_seconds:.6f} "
                    f"energy_increment_joules={energy_increment_joules:.6f} "
                    f"total_energy_joules={energy_joules[gpu.uuid]:.6f}"
                )

            previous_measurements = measurements

            if stop_event.wait(sample_interval_seconds):
                break

        LOGGER.debug(f"GPU energy measurement complete: {energy_joules}")

        result_queue.put(
            energy_joules,
            timeout=_QUEUE_INTERACTION_TIMEOUT,
        )

    except Exception as exception:  # pylint: disable=broad-except
        LOGGER.exception(f"GPU energy measurement failed: {exception}")

        result_queue.put(
            exception,
            timeout=_QUEUE_INTERACTION_TIMEOUT,
        )


class GPUPowerMeasurement(AbstractContextManager["GPUPowerMeasurement"]):
    """
    Measure GPU energy consumption for a context.

    :param gpus: GPUs to measure.
    :param sample_interval_seconds: Minimum delay between measurements.
    :param stop_event: Optional externally controlled stop event.
    """

    def __init__(
        self,
        gpus: Tuple[GPUIdentity, ...],
        sample_interval_seconds: float = _DEFAULT_SAMPLE_INTERVAL_SECONDS,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._gpus = gpus
        self._sample_interval_seconds = sample_interval_seconds

        self._stop_event = threading.Event() if stop_event is None else stop_event
        self._result_queue: queue.Queue[_MeasurementResult] = queue.Queue(
            maxsize=1,
        )
        self._thread: threading.Thread | None = None

        self._energy_joules: Dict[str, float] | None = None

    def __enter__(self) -> "GPUPowerMeasurement":
        """
        Start GPU power measurement.

        :return: The active measurement.
        :raises RuntimeError: If measurement has already started.
        """

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

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """
        Stop measurement and collect the result.

        :param exc_type: Exception type raised inside the context.
        :param exc_value: Exception raised inside the context.
        :param traceback: Traceback for the exception.
        :raises RuntimeError: If measurement was not started.
        """

        if self._thread is None:
            raise RuntimeError("GPU power measurement was never started")

        self._stop_event.set()
        self._thread.join()

        result = self._result_queue.get(
            timeout=_QUEUE_INTERACTION_TIMEOUT,
        )

        if isinstance(result, Exception):
            raise RuntimeError("GPU power measurement failed") from result

        self._energy_joules = result

    @property
    def energy_joules(self) -> Dict[str, float]:
        """
        Return measured energy consumption.

        :return: Energy in joules keyed by GPU UUID.
        :raises RuntimeError: If measurement has not completed.
        """

        if self._energy_joules is None:
            raise RuntimeError(
                "GPU power measurement results are not available until the "
                "measurement context has exited"
            )

        return self._energy_joules
