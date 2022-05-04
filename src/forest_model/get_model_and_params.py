from typing import Dict
from typing import List
from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from forest_model.consts import MODEL_PARAMS

Model = Union[RandomForestClassifier, LogisticRegression]


def get_model(model_name: str, random_state: int) -> Model:
    """Returns selected model by user's choice"""
    models: Dict[str, Model] = {
        "log": LogisticRegression(
            random_state=random_state, max_iter=300, solver="liblinear"
        ),
        "forest": RandomForestClassifier(random_state=random_state, n_estimators=100),
    }
    assert model_name in models, f"{model_name} is incorrect name!"
    return models[model_name]


def get_params(model_name: str, param_set: str) -> Dict[str, List[Union[str, int]]]:
    """Returns selected parameters by user's choice"""
    model_params = MODEL_PARAMS[model_name][param_set]
    return model_params  # type: ignore
