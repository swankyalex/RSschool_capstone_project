import argparse
import os
import pickle

import click
import numpy as np
from consts import DATA_PATH, DIR_MODEL
from get_data import get_data
from get_metrics import get_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        help="path to the train data",
        type=(str),
        default=os.path.join(DATA_PATH, "train.csv"),
    )
    parser.add_argument(
        "--s",
        help="path to save the model",
        type=(str),
        default=os.path.join(DIR_MODEL, "finalized_model.sav"),
    )
    parser.add_argument(
        "--random-state", help="set random state", type=(int), default=42
    )
    parser.add_argument(
        "--max-iter", help="set max iter count", type=(int), default=300
    )
    args = parser.parse_args()
    return vars(args).values()


data_path, model_path, random_state, max_iter = parse_args()
os.makedirs(os.path.dirname(model_path), exist_ok=True)


def train_model(data_path):
    X, y = get_data(data_path)
    model = LogisticRegression(
        solver="liblinear", C=0.01, max_iter=max_iter, random_state=random_state
    )
    scoring = get_metrics()
    scores = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
    accuracy = np.mean(scores["test_accuracy"])
    roc_auc = np.mean(scores["test_roc_auc"])
    f1 = np.mean(scores["test_f1"])
    log_loss = np.mean(scores["test_log_loss"])
    model.fit(X, y)
    pickle.dump(model, open(model_path, "wb"))
    click.echo(f"Model is saved to {model_path}.")
    click.echo(
        f"Result accuracy - {accuracy}, roc_auc - {roc_auc}, f1 score - {f1}, log_loss - {log_loss}"
    )


train_model(data_path)
