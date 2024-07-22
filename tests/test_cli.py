from pathlib import Path

import matplotlib
import pytest
from click.testing import CliRunner

from cfspopcon.cli import run_popcon_cli, write_algorithms_yaml


@pytest.mark.cli
@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_popcon_cli():
    matplotlib.use("Agg")

    runner = CliRunner()
    example_case = Path(__file__).parents[1] / "example_cases" / "SPARC_PRD"
    result = runner.invoke(
        run_popcon_cli,
        [str(example_case), "--show"],
    )
    assert result.exit_code == 0


@pytest.mark.cli
def test_write_algorithms_yaml(tmpdir):
    test_file = tmpdir.mkdir("test").join("test_popcon_algorithms.yaml")
    runner = CliRunner()
    result = runner.invoke(write_algorithms_yaml, ["-o", str(test_file)])
    assert result.exit_code == 0
    assert test_file.exists()
