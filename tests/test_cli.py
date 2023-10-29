from pathlib import Path

import matplotlib
import pytest
from click.testing import CliRunner

from cfspopcon.cli import run_popcon_cli


@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
def test_popcon_cli():
    matplotlib.use("Agg")

    runner = CliRunner()
    example_case = Path(__file__).parents[1] / "example_cases" / "SPARC_PRD"
    result = runner.invoke(
        run_popcon_cli,
        [str(example_case), "-p", str(example_case / "plot_popcon.yaml"), "-p", str(example_case / "plot_remapped.yaml"), "--show"],
    )
    assert result.exit_code == 0
