from typing import Any
from typing import Dict
from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

Model = Union[RandomForestClassifier, LogisticRegression]
models: Dict[str, Model] = {
    "log": LogisticRegression(max_iter=300, solver="liblinear"),
    "forest": RandomForestClassifier(n_estimators=100),
}

model_params: Dict[str, Dict[str, Any]] = {
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
            "max_depth": list(range(25, 36, 5)),
            "min_samples_leaf": list(range(2, 10, 2)),
            "min_samples_split": list(range(2, 10, 2)),
        },
    },
}
