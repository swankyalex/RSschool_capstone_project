import pandas as pd


def get_data(path: str):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y
