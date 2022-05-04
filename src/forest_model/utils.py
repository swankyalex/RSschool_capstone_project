import os
import pickle
from pathlib import Path

import click
import pandas as pd

from forest_model.model_settings import Model


def undummify(df: pd.DataFrame, prefix_sep: str = "_") -> pd.DataFrame:
    """Function to make back columns after get_dummies"""
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def save_model(model_path: Path, model_name: str, best_model: Model) -> None:
    path = os.path.join(model_path, f"{model_name}.sav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(best_model, open(path, "wb"))
    click.echo(f"Model is saved to {path}.")
