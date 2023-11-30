#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for cfspopcon."""
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import xarray as xr
from ipdb import launch_ipdb_on_exception  # type:ignore[import-untyped]

from cfspopcon import file_io
from cfspopcon.input_file_handling import read_case
from cfspopcon.plotting import make_plot, read_plot_style


@click.command()
@click.argument("case", type=click.Path(exists=True))
@click.option(
    "--plots",
    "-p",
    type=click.Path(exists=True),
    multiple=True,
)
@click.option("--show", is_flag=True, help="Display an interactive figure of the result")
@click.option("--debug", is_flag=True, help="Enable the ipdb exception catcher")
def run_popcon_cli(case: str, plots: tuple[str], show: bool, debug: bool) -> None:
    """Run POPCON from the command line.

    This function uses "Click" to develop the command line interface. You can execute it using
    poetry run python cfspopcon/cli.py --help

    You can specify a set of plots to create by specifying a plot style file after `-p` on the command-line. Multiple entries are supported.
    """
    if show and not plots:
        print(f"Speficied show={show}, but did not specify a plot style, see --plots!")
        sys.exit(1)

    if not debug:
        run_popcon(case, plots, show)
    else:
        with launch_ipdb_on_exception():
            run_popcon(case, plots, show)


def run_popcon(case: str, plots: tuple[str], show: bool) -> None:
    """Run popcon case.

    Args:
        case: specify case to run (corresponding to a case in cases)
        plots: specify which plots to make (corresponding to a plot_style in plot_styles)
        show: show the resulting plots
    """
    input_parameters, algorithm, points = read_case(case)

    dataset = xr.Dataset(input_parameters)

    algorithm.validate_inputs(dataset)
    algorithm.update_dataset(dataset, in_place=True)

    output_dir = Path(case) / "output" if Path(case).is_dir() else Path(case).parent / "output"
    output_dir.mkdir(exist_ok=True)

    file_io.write_dataset_to_netcdf(dataset, filepath=output_dir / "dataset.nc")

    for point, point_params in points.items():
        file_io.write_point_to_file(dataset, point, point_params, output_dir=output_dir)

    # Plot the results
    for plot_style in plots:
        make_plot(dataset, read_plot_style(plot_style), points, title=input_parameters.get("plot_title", "POPCON"), output_dir=output_dir)

    print("Done")
    if show:
        plt.show()


if __name__ == "__main__":
    run_popcon_cli()
