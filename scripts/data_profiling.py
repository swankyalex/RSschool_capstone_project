import argparse
import os

import pandas as pd
from pandas_profiling import ProfileReport

from forest_model.consts import DATA_PATH


def parse_args() -> str:
    """Parse args from CLI
    --d Path to the data"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        help="path to the data",
        type=str,
        default=os.path.join(DATA_PATH, "train.csv"),
    )
    args = parser.parse_args()
    return args.d


def pandas_profiling() -> None:
    """Making pandas profile report"""
    data = pd.read_csv(data_path)
    profile = ProfileReport(data, title="Pandas Profiling Report", minimal=True)
    profile.to_file(os.path.join(DATA_PATH, "forest_report.html"))


if __name__ == "__main__":
    data_path = parse_args()
    pandas_profiling()
