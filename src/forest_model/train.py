import argparse
import os
import pickle

import click
import mlflow
import numpy as np
from consts import DATA_PATH, DIR_MODEL
from get_data import get_data
from get_metrics import get_metrics
from get_model_and_params import get_model, get_params
from sklearn.model_selection import GridSearchCV, cross_validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        help="path to the train data",
        type=str,
        default=os.path.join(DATA_PATH, "train.csv"),
    )
    parser.add_argument(
        "--s",
        help="path to save the model",
        type=str,
        default=os.path.join(DIR_MODEL, "finalized_model.sav"),
    )
    parser.add_argument("--random-state", help="set random state", type=int, default=42)
    parser.add_argument(
        "--model",
        help="Choose model: <log> for log regression, <forest> for random forest",
        type=str,
        default="log",
    )
    parser.add_argument(
        "--params", help="Choose parameters set: [1,2,3]", type=str, default="1"
    )
    args = parser.parse_args()
    return vars(args).values()


data_path, model_path, random_state, model_name, parameter_set = parse_args()
os.makedirs(os.path.dirname(model_path), exist_ok=True)


def train_model():
    X, y = get_data(data_path)
    scoring = get_metrics()
    model = get_model(model_name, random_state)
    params = get_params(model_name, parameter_set)
    gsc = GridSearchCV(model, params, cv=5, scoring="accuracy", refit=True, n_jobs=-1)
    gsc.fit(X, y)
    best_model = gsc.best_estimator_
    best_params = gsc.best_params_
    with mlflow.start_run():
        scores = cross_validate(best_model, X, y, cv=5, scoring=scoring, n_jobs=-1)
        accuracy = np.mean(scores["test_accuracy"])
        roc_auc = np.mean(scores["test_roc_auc"])
        f1 = np.mean(scores["test_f1"])
        log_loss = np.mean(scores["test_log_loss"])
        mlflow.log_param("Model", model_name)
        for param_name, value in best_params.items():
            mlflow.log_param(param_name, value)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("log_loss", float(log_loss))
        best_model.fit(X, y)
        pickle.dump(best_model, open(model_path, "wb"))
        click.echo(f"Model is saved to {model_path}.")
        click.echo(f"Result accuracy - {accuracy}")


if __name__ == "__main__":
    train_model()
