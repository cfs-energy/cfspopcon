"""Methods to run analyses configured via input files."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import xarray as xr
import yaml

from .algorithm_class import Algorithm, CompositeAlgorithm
from .helpers import convert_named_options
from .unit_handling import set_default_units


def read_case(
    case: Union[str, Path, dict[str, Any]], kwargs: Optional[dict[str, str]] = None
) -> tuple[dict[str, Any], Union[CompositeAlgorithm, Algorithm], dict[str, Any], dict[str, Path]]:
    """Read a yaml file corresponding to a given case.

    case should be passed either as a complete filepath to an input.yaml file or to
    the parent folder of an input.yaml file.

    kwargs can be an arbitrary dictionary of key-value pairs that overwrite the config values.
    """
    if kwargs is None:
        kwargs = dict()

    if isinstance(case, dict):
        repr_d = case
        case_dir = Path(".").absolute()

    elif Path(case).exists():
        case = Path(case)
        if case.is_dir():
            case_dir = case
            input_file = case_dir / "input.yaml"
        else:
            case_dir = case.parent
            input_file = case

        with open(input_file) as file:
            repr_d = yaml.load(file, Loader=yaml.FullLoader)

    else:
        raise FileNotFoundError(f"Could not find {case}.")

    repr_d.update(kwargs)

    algorithms = repr_d.pop("algorithms", dict())
    algorithm_list: list[Union[Algorithm, CompositeAlgorithm]] = [Algorithm.get_algorithm(algorithm) for algorithm in algorithms]

    if len(algorithm_list) > 1:
        algorithm = CompositeAlgorithm(algorithm_list)
    elif len(algorithm_list) == 1:
        algorithm = algorithm_list[0]  # type:ignore[assignment]
    elif len(algorithm_list) == 0:
        algorithm = Algorithm.empty()  # type:ignore[assignment]

    points = repr_d.pop("points", dict())
    plots = repr_d.pop("plots", dict())

    process_grid_values(repr_d)
    process_named_options(repr_d)
    process_paths(repr_d, case_dir)
    process_paths(plots, case_dir)
    process_units(repr_d)

    return repr_d, algorithm, points, plots


def process_grid_values(repr_d: dict[str, Any]):  # type:ignore[no-untyped-def]
    """Process the grid of values to run POPCON over."""
    grid_values = repr_d.pop("grid", dict())
    for key, grid_spec in grid_values.items():
        grid_spacing = grid_spec.get("spacing", "linear")

        if grid_spacing == "linear":
            grid_vals = np.linspace(grid_spec["min"], grid_spec["max"], num=grid_spec["num"])
        elif grid_spacing == "log":
            grid_vals = np.logspace(np.log10(grid_spec["min"]), np.log10(grid_spec["max"]), num=grid_spec["num"])
        else:
            raise NotImplementedError(f"No implementation for grid with {grid_spec['spacing']} spacing.")

        repr_d[key] = xr.DataArray(grid_vals, coords={f"dim_{key}": grid_vals})


def process_named_options(repr_d: dict[str, Any]):  # type:ignore[no-untyped-def]
    """Process named options (enums), handling also list arguments."""
    for key, val in repr_d.items():
        if isinstance(val, (list, tuple)):
            repr_d[key] = [convert_named_options(key=key, val=v) for v in val]
        else:
            repr_d[key] = convert_named_options(key=key, val=val)


def process_paths(repr_d: dict[str, Any], case_dir: Path):  # type:ignore[no-untyped-def]
    """Process path tags, up to a maximum of one tag per input variable.

    Allowed tags are:
    * CASE_DIR: the folder that the input.yaml file is located in
    * WORKING_DIR: the current working directory that the script is being run from
    """
    path_mappings = dict(
        CASE_DIR=case_dir,
        WORKING_DIR=Path("."),
    )
    if repr_d is None:
        return

    for key, val in repr_d.items():
        if isinstance(val, str):
            for replace_key, replace_path in path_mappings.items():
                if replace_key in val:
                    path_val = Path(val.replace(replace_key, str(replace_path.absolute()))).absolute()
                    repr_d[key] = path_val
                    break


def process_units(repr_d: dict[str, Any]):  # type:ignore[no-untyped-def]
    """Set default units on each of the input variables."""
    for key, val in repr_d.items():
        repr_d[key] = set_default_units(key=key, value=val)
