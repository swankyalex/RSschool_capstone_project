import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utils import undummify


def process_data(df, process_type):
    processing_types = {"1": processing_1, "2": processing_2}
    df = processing_types[process_type](df)
    return df


def processing_1(df):
    cat_data = undummify(df.iloc[:, 10:])
    num_data = df.iloc[:, :10]
    df = pd.concat([num_data, cat_data], axis=1)
    wilderness = {"Area1": 1, "Area2": 2, "Area3": 3, "Area4": 4}
    df["Wilderness"] = df["Wilderness"].map(wilderness)
    encoder = LabelEncoder()
    df.iloc[:, -1] = encoder.fit_transform(df.iloc[:, -1])
    df = StandardScaler().fit_transform(df)
    return df


def processing_2(df):
    cat_data = undummify(df.iloc[:, 10:])
    num_data = df.iloc[:, :10]
    df = pd.concat([num_data, cat_data], axis=1)
    wilderness = {"Area1": 1, "Area2": 2, "Area3": 3, "Area4": 4}
    df["Wilderness"] = df["Wilderness"].map(wilderness)
    df.drop(columns=["Soil"], axis=1, inplace=True)
    df = StandardScaler().fit_transform(df)
    return df
