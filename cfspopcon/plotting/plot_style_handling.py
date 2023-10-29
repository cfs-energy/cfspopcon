"""Handling of yaml plot configuration."""
from pathlib import Path
from typing import Any, Union

import yaml


def read_plot_style(plot_style: Union[str, Path]) -> dict[str, Any]:
    """Read a yaml file corresponding to a given plot_style.

    plot_style may be passed either as a complete filepath or as a string matching a plot_style in "plot_styles"
    """
    if Path(plot_style).exists():
        input_file = plot_style
    else:
        raise FileNotFoundError(f"Could not find {plot_style}!")

    with open(input_file) as file:
        repr_d: dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    return repr_d
