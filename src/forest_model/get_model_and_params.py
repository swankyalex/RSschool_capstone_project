from typing import Dict
from typing import List
from typing import Union

from forest_model.model_settings import Model
from forest_model.model_settings import model_params
from forest_model.model_settings import models


def get_model(model_name: str, random_state: int) -> Model:
    """Returns selected model by user's choice"""
    assert model_name in models, f"{model_name} is incorrect name!"
    model = models[model_name]
    model.random_state = random_state
    return model


def get_params(model_name: str, param_set: str) -> Dict[str, List[Union[str, int]]]:
    """Returns selected parameters by user's choice"""
    params = model_params[model_name][param_set]
    return params  # type: ignore
