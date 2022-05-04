from typing import Dict
from typing import List
from typing import Union

from forest_model.consts import Model
from forest_model.consts import MODEL_PARAMS
from forest_model.consts import MODELS


def get_model(model_name: str, random_state: int) -> Model:
    """Returns selected model by user's choice"""

    assert model_name in MODELS, f"{model_name} is incorrect name!"
    model = MODELS[model_name]
    model.random_state = random_state
    return model


def get_params(model_name: str, param_set: str) -> Dict[str, List[Union[str, int]]]:
    """Returns selected parameters by user's choice"""
    model_params = MODEL_PARAMS[model_name][param_set]
    return model_params  # type: ignore
