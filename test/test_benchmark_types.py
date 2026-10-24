"""
Tests to make sure the type system and its extensions are valid.
"""

from gpu_box_benchmark.benchmark_types import (
    BenchmarkName,
    BenchmarkFamily,
    BENCHMARK_TO_PRETTY,
    EXTENDED_BENCHMARK_DOCUMENTS,
)


def test_lookups_complete() -> None:
    """
    There are a few Enum -> Docs lookups. Check if there are any missing members here.
    :return: None
    """

    assert all(member in BENCHMARK_TO_PRETTY for member in list(BenchmarkName))
    assert all(member in EXTENDED_BENCHMARK_DOCUMENTS for member in list(BenchmarkFamily))
