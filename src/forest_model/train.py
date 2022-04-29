import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from forest_model.get_data import get_data
from scripts.consts import DATA_PATH


def train_model(data_path):
    X, y = get_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(accuracy)


path = os.path.join(DATA_PATH, "train.csv")
train_model(path)
