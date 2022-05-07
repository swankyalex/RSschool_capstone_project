import os
import pickle

import click
import pandas as pd

from forest_model.consts import DATA_PATH
from forest_model.consts import DIR_MODEL
from forest_model.data_processing import process_data
from forest_model.model_settings import model_params


@click.command()
@click.option(
    "--model-name",
    default="forest",
    type=click.Choice(list(model_params.keys()), case_sensitive=False),
    help=f"Choose model: {list(model_params.keys())}",
    show_default=True,
)
def make_submission(model_name: str) -> None:
    model = pickle.load(open(os.path.join(DIR_MODEL, f"{model_name}.sav"), "rb"))
    test_data = os.path.join(DATA_PATH, "test.csv")
    X_test = pd.read_csv(test_data, index_col="Id")
    X_test = process_data(X_test, process_type="1")
    predictions = model.predict(X_test)
    submit = pd.DataFrame(data={"Id": X_test.index, "Cover_Type": predictions})
    path = os.path.join(DATA_PATH, "submission.csv")
    submit.to_csv(path, index=False)
    click.echo(f"submission is saved to {path}.")


if __name__ == "__main__":
    make_submission()
