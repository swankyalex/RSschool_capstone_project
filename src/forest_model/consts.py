from pathlib import Path
from typing import Any
from typing import Dict

_this_file = Path(__file__).resolve()

DIR_REPO = _this_file.parent.parent.parent.resolve()
DATA_PATH = (DIR_REPO / "data").resolve()
DIR_SCRIPTS = (DIR_REPO / "scripts").resolve()
DIR_SRC = (DIR_REPO / "src").resolve()
DIR_MODEL = (DIR_REPO / "models").resolve()
DIR_FIXTURES = (DIR_REPO / "tests" / "fixtures").resolve()
MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "log": {
        "1": {"C": [0.01, 0.1, 1], "fit_intercept": [True, False]},
        "2": {"C": [0.0001, 0.001, 0.01], "fit_intercept": [True, False]},
        "3": {"C": [1, 10, 100], "fit_intercept": [True, False]},
    },
    "forest": {
        "1": {
            "max_depth": [5, 7, 9],
            "min_samples_leaf": [2, 5, 7],
            "min_samples_split": [2, 5, 9],
        },
        "2": {
            "max_depth": [3, 4, 5],
            "min_samples_leaf": [1, 2, 3],
            "min_samples_split": [2, 3, 4],
        },
        "3": {
            "max_depth": [8, 9, 10],
            "min_samples_leaf": [10, 20, 30],
            "min_samples_split": [10, 15, 20],
        },
    },
}
