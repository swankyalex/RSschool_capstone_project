from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

Model = Union[RandomForestClassifier, LogisticRegression]

models = {"log": LogisticRegression, "forest": RandomForestClassifier}

log_params = {
    "1": {"C": [0.01, 0.1, 1], "fit_intercept": [True, False]},
    "2": {"C": [0.0001, 0.001, 0.01], "fit_intercept": [True, False]},
    "3": {"C": [1, 10, 100], "fit_intercept": [True, False]},
}

forest_params = {
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
}

params = {"log": log_params, "forest": forest_params}


def get_model(model_name: str, random_state: int) -> Model:
    """Returns selected model by user's choice"""
    assert model_name in models, f"{model_name} is incorrect name!"
    model = models[model_name]
    if model_name == "log":
        model = model(random_state=random_state, max_iter=300, solver="liblinear")
    elif model_name == "forest":
        model = model(random_state=random_state, n_estimators=100)
    return model


def get_params(model_name: str, param_set: str) -> dict:
    """Returns selected parameters by user's choice"""
    parameters = params[model_name][param_set]
    return parameters
