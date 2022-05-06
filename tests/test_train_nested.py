from evaluation_selection import __version__


from click.testing import CliRunner
import pytest

from evaluation_selection.train_nested import train

def test_version():
    assert __version__ == '0.1.0'


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


def test_for_valid_knested(runner: CliRunner) -> None:
    """It valid when test knested is equal 1."""
    result = runner.invoke(
        train,
        [
            "--knested",
            1,
        ],
    )
    assert result.exit_code == 1