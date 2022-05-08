from evaluation_selection import __version__


from click.testing import CliRunner
import pytest

from evaluation_selection.train_nested import train
import pandas as pd
import numpy as np
from sklearn.datasets import                                                                                               make_multilabel_classification

X, y = make_multilabel_classification(random_state=42)
df = pd.DataFrame(np.c_[X, y.sum(axis=1)])


def test_version() -> None:
    assert __version__ == "0.1.0"


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_kfold(runner: CliRunner) -> None:
    """It fails when test kfold is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--kfold",
            "str",
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--kfold'" in result.output


def test_for_valid_knested(runner: CliRunner) -> None:
    """It valid when test knested is equal 1."""
    with runner.isolated_filesystem():
        df.to_csv("test_random.csv")
        result = runner.invoke(
            train,
            [
                "--dataset-path",
                "test_random.csv",
                "--save-model-path",
                "test_model.joblib",
                "--knested",
                "1",
            ],
        )
    assert result.exit_code == 1
