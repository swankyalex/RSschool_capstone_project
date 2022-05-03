from pathlib import Path
from typing import Tuple

import pandas as pd

from forest_model.data_processing import process_data


def get_train_data(path: Path, process_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns splitted processed train data"""
    data = pd.read_csv(path, index_col="Id")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = process_data(X, process_type)
    return X, y
