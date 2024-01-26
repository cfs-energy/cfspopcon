"""Methods to run analyses configured via input files."""

from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr
import yaml

from .algorithms import get_algorithm
from .algorithms.algorithm_class import Algorithm, CompositeAlgorithm
from .helpers import convert_named_options
from .unit_handling import set_default_units


def read_case(case: Union[str, Path]) -> tuple[dict[str, Any], Union[CompositeAlgorithm, Algorithm], dict[str, Any]]:
    """Read a yaml file corresponding to a given case.

    case should be passed either as a complete filepath to an input.yaml file or to
    the parent folder of an input.yaml file.
    """
    if Path(case).exists():
        case = Path(case)
        if case.is_dir():
            input_file = case / "input.yaml"
        else:
            input_file = case
    else:
        raise FileNotFoundError(f"Could not find {case}.")

    with open(input_file) as file:
        repr_d: dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    algorithms = repr_d.pop("algorithms")
    algorithm_list = [get_algorithm(algorithm) for algorithm in algorithms]

    # why doesn't mypy deduce the below without hint?
    algorithm: Union[Algorithm, CompositeAlgorithm] = CompositeAlgorithm(algorithm_list) if len(algorithm_list) > 1 else algorithm_list[0]

    points = repr_d.pop("points")

    grid_values = repr_d.pop("grid")

    for key, grid_spec in grid_values.items():
        grid_spacing = grid_spec.get("spacing", "linear")

        if grid_spacing == "linear":
            grid_vals = np.linspace(grid_spec["min"], grid_spec["max"], num=grid_spec["num"])
        elif grid_spacing == "log":
            grid_vals = np.logspace(np.log10(grid_spec["min"]), np.log10(grid_spec["max"]), num=grid_spec["num"])
        else:
            raise NotImplementedError(f"No implementation for grid with {grid_spec['spacing']} spacing.")

        repr_d[key] = xr.DataArray(grid_vals, coords={f"dim_{key}": grid_vals})

    for key, val in repr_d.items():
        if isinstance(val, (list, tuple)):
            repr_d[key] = [convert_named_options(key=key, val=v) for v in val]
        else:
            repr_d[key] = convert_named_options(key=key, val=val)

    for key, val in repr_d.items():
        repr_d[key] = set_default_units(key=key, value=val)

    return repr_d, algorithm, points
