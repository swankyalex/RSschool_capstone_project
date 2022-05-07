import os
import pickle

import numpy as np
import pytest
from click.testing import CliRunner
from sklearn.model_selection import cross_val_score

from forest_model.consts import DIR_FIXTURES
from forest_model.get_data import get_train_data
from forest_model.train import train_model


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_random_state(runner: CliRunner) -> None:
    """It fails when random state is not int."""
    result = runner.invoke(
        train_model,
        [
            "--random-state",
            "forty_two",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output


def test_error_for_invalid_model(runner: CliRunner) -> None:
    """It fails when test model name is not available"""
    result = runner.invoke(
        train_model,
        [
            "--model-name",
            "Restnet",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--model-name'" in result.output


def test_train_function(runner: CliRunner, tmp_path) -> None:
    """Testing the model on some small sample of data, check it for correctness saving,
    checking test accuracy in correct range,and data has no duplicates or none values"""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            train_model,
            [
                "--data-path",
                os.path.join(DIR_FIXTURES, "test.csv"),
                "--model-path",
                tmp_path,
            ],
        )

        model = pickle.load(open(os.path.join(tmp_path, "log.sav"), "rb"))
        X, y = get_train_data(os.path.join(DIR_FIXTURES, "test.csv"), "1")
        scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
        accuracy = float(np.mean(scores))

        assert 0.5 < accuracy <= 1
        assert X.duplicated().sum() == 0
        assert X.isna().sum().all() == 0
        assert result.exit_code == 0
