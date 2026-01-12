"""
Code for running the "nvidia deep learning examples" benchmarks and parsing the output.
"""

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class ReportFileNumerical(NamedTuple):
    """
    Intermediate type to contain the numerical result.
    """

    sample_count: float
    mean: float
    std: float
    result_min: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    result_max: float


def parse_report_file(report_path: Path) -> ReportFileNumerical:
    """
    Parse a report file to the standard set of numerical results.
    :param report_path: Path to the report file on disk.
    :return: Numerical results
    """

    with open(report_path, encoding="utf-8", mode="r") as report_file:
        loaded_dicts = [json.loads(line.replace("DLLL ", "")) for line in report_file.readlines()]
        complete_df = pd.DataFrame.from_records(loaded_dicts)

    data_only_df = pd.DataFrame.from_records(complete_df[complete_df["type"] == "LOG"]["data"])
    summary_dict = data_only_df["train.total_ips"].dropna().describe().to_dict()

    output = ReportFileNumerical(
        sample_count=summary_dict["count"],
        mean=summary_dict["mean"],
        std=summary_dict["std"],
        result_min=summary_dict["min"],
        percentile_25=summary_dict["25%"],
        percentile_50=summary_dict["50%"],
        percentile_75=summary_dict["75%"],
        result_max=summary_dict["max"],
    )

    return output
