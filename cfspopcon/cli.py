#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for cfspopcon."""
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
@click.option("--show", is_flag=True, help="Display an interactive figure of the result")
@click.option("--debug", is_flag=True, help="Enable the ipdb exception catcher")
def run_popcon_cli(case: str, show: bool, debug: bool) -> None:
    """Run POPCON from the command line.

    This function uses "Click" to develop the command line interface. You can execute it using
    poetry run python cfspopcon/cli.py --help
    """
    if not debug:
        run_popcon(case, show)
    else:
        with launch_ipdb_on_exception():
            run_popcon(case, show)


def run_popcon(case: str, show: bool) -> None:
    """Run popcon case.

    Args:
        case: specify case to run (corresponding to a case in cases)
        show: show the resulting plots
    """
    input_parameters, algorithm, points, plots = read_case(case)

    dataset = xr.Dataset(input_parameters)

    algorithm.validate_inputs(dataset)
    dataset = algorithm.update_dataset(dataset)

    output_dir = Path(case) / "output" if Path(case).is_dir() else Path(case).parent / "output"
    output_dir.mkdir(exist_ok=True)

    file_io.write_dataset_to_netcdf(dataset, filepath=output_dir / "dataset.nc")

    for point, point_params in points.items():
        file_io.write_point_to_file(dataset, point, point_params, output_dir=output_dir)

    # Plot the results
    for plot_name, plot_style in plots.items():
        print(f"Plotting {plot_name}")
        make_plot(dataset, read_plot_style(plot_style), points, title=plot_name, output_dir=output_dir, save_name=plot_style.stem)

    print("Done")
    if show:
        plt.show()


if __name__ == "__main__":
    run_popcon_cli()
