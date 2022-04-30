import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, make_scorer


def get_metrics():
    accuracy = make_scorer(accuracy_score)
    f1 = make_scorer(f1_score, average="weighted")
    ll = make_scorer(log_loss, needs_proba=True)
    scoring = {"accuracy": accuracy, "f1": f1, "roc_auc": "roc_auc_ovr", "log_loss": ll}
    return scoring
