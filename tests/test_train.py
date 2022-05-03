from click.testing import CliRunner
import pytest

from forest_model.train import train_model


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_random_state(
        runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train_model,
        [
            "--random-state",
            "forty_two",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output


def test_error_for_invalid_ьщвуд(
        runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train_model,
        [
            "--model-name",
            "Restnet",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--model-name'" in result.output
