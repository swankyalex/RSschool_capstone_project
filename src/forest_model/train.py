import os
from pathlib import Path
from typing import Tuple
from typing import Union

import click
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from forest_model.consts import DATA_PATH
from forest_model.consts import DIR_MODEL
from forest_model.data_processing import processing_types
from forest_model.get_data import get_train_data
from forest_model.get_metrics import get_metrics
from forest_model.get_model_and_params import get_model
from forest_model.get_model_and_params import get_params
from forest_model.model_settings import Model
from forest_model.model_settings import model_params
from forest_model.utils import save_model


@click.command()
@click.option(
    "-d",
    "--data-path",
    default=os.path.join(DATA_PATH, "train.csv"),
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--model-path",
    default=DIR_MODEL,
    type=click.Path(exists=False, dir_okay=True, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--model-name",
    default="log",
    type=click.Choice(list(model_params.keys()), case_sensitive=False),
    help=f"Choose model: {list(model_params.keys())}",
    show_default=True,
)
@click.option(
    "--params",
    default="1",
    type=click.Choice(list(model_params["log"].keys())),
    help=f"Choose parameters set: {list(model_params['log'].keys())}",
    show_default=True,
)
@click.option(
    "--evaluate",
    default=True,
    type=bool,
    help="Evaluation needed (True, False)",
    show_default=True,
)
@click.option(
    "--proc-type",
    default="1",
    type=click.Choice(list(processing_types.keys())),
    help=f"Choose processing type: {list(processing_types.keys())}",
    show_default=True,
)
def train_model(
    data_path: Path,
    model_path: Path,
    random_state: int,
    model_name: str,
    params: str,
    evaluate: bool,
    proc_type: str,
) -> None:
    """Training the model"""
    X, y = get_train_data(data_path, proc_type)
    model = get_model(model_name, random_state)
    param = get_params(model_name, params)
    if evaluate:
        best_model, accuracy = train_and_evaluate(
            X, y, model, param, random_state, model_name, proc_type
        )
    else:
        best_model, accuracy = train_without_eval(X, y, model, param)
    save_model(model_path, model_name, best_model)
    click.echo(f"Result accuracy - {accuracy}")


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model: Model,
    params: dict[str, list[Union[str, int]]],
    random_state: int,
    model_name: str,
    proc_type: str,
) -> Tuple[Model, float]:
    """Train model with NestedCV validation and logging parameters to ML flow"""
    scoring = get_metrics()
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
    gsc = GridSearchCV(
        model, params, cv=cv_inner, scoring="accuracy", refit=True, n_jobs=-1
    )
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
    with mlflow.start_run():
        scores = cross_validate(gsc, X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
        accuracy = float(np.mean(scores["test_accuracy"]))
        roc_auc = np.mean(scores["test_roc_auc"])
        f1 = np.mean(scores["test_f1"])
        log_loss = np.mean(scores["test_log_loss"])
        mlflow.log_param("Model", model_name)
        mlflow.log_param("Processing type", proc_type)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("log_loss", float(log_loss))
        gsc.fit(X, y)
        best_model = gsc.best_estimator_
        best_params = gsc.best_params_
        for param_name, value in best_params.items():
            mlflow.log_param(param_name, value)
    return best_model, accuracy


def train_without_eval(
    X: pd.DataFrame,
    y: pd.Series,
    model: Model,
    params: dict[str, list[Union[str, int]]],
) -> Tuple[Model, float]:
    """Train model without NestedCV validation"""
    gsc = GridSearchCV(model, params, cv=5, scoring="accuracy", refit=True, n_jobs=-1)
    gsc.fit(X, y)
    best_model = gsc.best_estimator_
    accuracy = gsc.best_score_
    return best_model, accuracy


if __name__ == "__main__":
    train_model()
