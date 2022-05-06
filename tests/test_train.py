from evaluation_selection import __version__

import click
from click.testing import CliRunner
import pytest
import pandas as pd
import numpy as np
from joblib import load

from evaluation_selection.train import train
from evaluation_selection.data import get_dataset

def test_version():
    assert __version__ == '0.1.0'


@click.command()
@click.argument('f')
def load_test(f):
   click.echo(load(f))

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
            'str',
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--kfold'" in result.output

def test_for_valid_kfold(runner: CliRunner) -> None:
    """It fails when test kfold is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--kfold",
            3,
        ],
    )
    assert result.exit_code == 0


def test_for_save_model_with_isolated_filesystem(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        from sklearn.datasets import make_multilabel_classification
        X, y = make_multilabel_classification(random_state=42)
        df = pd.DataFrame(np.c_[X, y.sum(axis=1)])
        df.to_csv('test_random.csv')
        X_train, X_val, y_train, y_val = get_dataset('test_random.csv', test_size=0.1, random_state=42)
        assert [X_train.shape, X_val.shape, y_train.shape, y_val.shape] == [(90, 20), (10, 20), (90,), (10,)]

        runner.invoke(
            train,
            [
                "--dataset-path",
                'test_random.csv',
                "--save-model-path",
                'test_model.joblib',
            ],
        )

        result = runner.invoke(load_test, ['test_model.joblib'])
        assert result.exit_code == 0
        assert result.output.replace(' ', '').replace('\n', '') \
               == "Pipeline(steps=[('threshold',VarianceThreshold(threshold=0.01))," \
                  "('classifier',KNeighborsClassifier(n_jobs=-1))])"
