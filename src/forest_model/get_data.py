import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data(path: str):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = StandardScaler().fit_transform(X)
    return X, y
