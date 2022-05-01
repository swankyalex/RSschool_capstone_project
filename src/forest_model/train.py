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
from sklearn.model_selection import GridSearchCV, KFold, cross_validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        help="path to the train data",
        type=str,
        default=os.path.join(DATA_PATH, "train.csv"),
    )
    parser.add_argument(
        "--s", help="path to save the model", type=str, default=DIR_MODEL
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
    parser.add_argument(
        "--evaluate", help="Evaluation needed (0, 1)", type=bool, default=False
    )
    args = parser.parse_args()
    return vars(args).values()


data_path, model_path, random_state, model_name, parameter_set, evaluate = parse_args()
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model_path = os.path.join(DIR_MODEL, f"{model_name}.sav")


def train_and_evaluate(X, y, model, params):
    scoring = get_metrics()
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
    gsc = GridSearchCV(
        model, params, cv=cv_inner, scoring="accuracy", refit=True, n_jobs=-1
    )
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
    with mlflow.start_run():
        scores = cross_validate(gsc, X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
        accuracy = np.mean(scores["test_accuracy"])
        roc_auc = np.mean(scores["test_roc_auc"])
        f1 = np.mean(scores["test_f1"])
        log_loss = np.mean(scores["test_log_loss"])
        mlflow.log_param("Model", model_name)
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


def train_without_eval(X, y, model, params):
    gsc = GridSearchCV(model, params, cv=5, scoring="accuracy", refit=True, n_jobs=-1)
    gsc.fit(X, y)
    best_model = gsc.best_estimator_
    accuracy = gsc.best_score_
    return best_model, accuracy


def train_model():
    X, y = get_data(data_path)
    model = get_model(model_name, random_state)
    params = get_params(model_name, parameter_set)
    if evaluate:
        best_model, accuracy = train_and_evaluate(X, y, model, params)
    else:
        best_model, accuracy = train_without_eval(X, y, model, params)
    pickle.dump(best_model, open(model_path, "wb"))
    click.echo(f"Model is saved to {model_path}.")
    click.echo(f"Result accuracy - {accuracy}")


if __name__ == "__main__":
    train_model()
