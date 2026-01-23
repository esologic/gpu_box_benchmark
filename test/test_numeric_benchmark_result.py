"""
Smoke tests for now.
"""

import tempfile
from pathlib import Path

from gpu_box_benchmark import numeric_benchmark_result
from test.assets import ALL_SAMPLE_OUTPUTS

SAMPLE_SYSTEM_EVALUATION = """
{
  "title": "Automated GBB Run",
  "description": "Pollicaris GPU 1 Development VM, working on getting everything in place for a bigass set of runs.",
  "gpu_box_benchmark_version": "0.1.0",
  "cpu": {
    "name": "Intel(R) Xeon(R) CPU E5-2687W v4 @ 3.00GHz",
    "physical_cpus": 48,
    "logical_cpus": 48
  },
  "total_memory_gb": 95.91012573242188,
  "gpus": [
    {
      "id": 0,
      "name": "Tesla P100-PCIE-16GB",
      "uuid": "GPU-6054226f-b047-d1d8-3ac1-61d0168eba23",
      "total_memory_mib": 16384.0,
      "driver": "550.163.01",
      "serial": "0324616116378"
    },
    {
      "id": 1,
      "name": "Tesla V100-PCIE-16GB",
      "uuid": "GPU-487d9068-bb38-14d4-ce6f-f77ef379bbc7",
      "total_memory_mib": 16384.0,
      "driver": "550.163.01",
      "serial": "0320618047538"
    },
    {
      "id": 2,
      "name": "Tesla P100-PCIE-16GB",
      "uuid": "GPU-2770b63a-fd9e-03fc-f0e7-d1afb5af794f",
      "total_memory_mib": 16384.0,
      "driver": "550.163.01",
      "serial": "0321017108565"
    },
    {
      "id": 3,
      "name": "Tesla P100-PCIE-16GB",
      "uuid": "GPU-68b7cf03-5caf-6e6a-05af-ffea91884299",
      "total_memory_mib": 16384.0,
      "driver": "550.163.01",
      "serial": "0323117070701"
    }
  ],
  "start_time": "2026-01-20T21:29:43.424875",
  "runtime_seconds": 11652.396897,
  "results": [
    {
      "name": "resnet50_train_batch_1_amp",
      "benchmark_version": "0.1.0",
      "override_parameters": {},
      "larger_better": true,
      "multi_gpu_native": true,
      "verbose_unit": "Images Processed / Second",
      "unit": "i/s",
      "critical_result_key": "native_multi_gpu_result",
      "numerical_results": {
        "min_by_gpu_type": 31.552277507270514,
        "max_by_gpu_type": 33.491155606653386,
        "mean_by_gpu_type": 32.52171655696195,
        "theoretical_multi_gpu_mean": 32.03699703211623,
        "theoretical_multi_gpu_sum": 128.14798812846493,
        "forced_multi_gpu_numerical_mean": 30.81416508213364,
        "forced_multi_gpu_sum": 123.25666032853456,
        "native_multi_gpu_result": 61
      }
    },
    {
      "name": "resnet50_train_batch_2_amp",
      "benchmark_version": "0.1.0",
      "override_parameters": {},
      "larger_better": true,
      "multi_gpu_native": true,
      "verbose_unit": "Images Processed / Second",
      "unit": "i/s",
      "critical_result_key": "native_multi_gpu_result",
      "numerical_results": {
        "min_by_gpu_type": 31.552277507270514,
        "max_by_gpu_type": 33.491155606653386,
        "mean_by_gpu_type": 32.52171655696195,
        "theoretical_multi_gpu_mean": 32.03699703211623,
        "theoretical_multi_gpu_sum": 128.14798812846493,
        "forced_multi_gpu_numerical_mean": 30.81416508213364,
        "forced_multi_gpu_sum": 123.25666032853456,
        "native_multi_gpu_result": 61
      }
    }
  ]
}
"""


def test_load_system_evaluation_from_disk() -> None:
    """
    Test to make sure loading a known good string works.
    :return: None
    """

    with tempfile.NamedTemporaryFile(mode="w") as file:
        file.write(SAMPLE_SYSTEM_EVALUATION)
        file.seek(0)
        loaded = numeric_benchmark_result.load_system_evaluation_from_disk(path=Path(file.name))

    assert loaded.results[0].numerical_results.native_multi_gpu_result == 61


def test_known_valid_loads() -> None:
    """
    Try loading all the known good test assets.
    :return: None
    """

    for path in ALL_SAMPLE_OUTPUTS:
        numeric_benchmark_result.load_system_evaluation_from_disk(path=path)
