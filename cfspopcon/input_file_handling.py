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
from freeqdsk import geqdsk
import scipy.interpolate

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

    #Variables to be extracted from the equilibrium file
    key_list = [
        "major_radius",
        "magnetic_field_on_axis",
        "inverse_aspect_ratio",
        "areal_elongation",
        "elongation_ratio_sep_to_areal",
        "triangularity_psi95",
        "triangularity_ratio_sep_to_psi95",
        "plasma_current"
        ]
    
    if "equilibrium_file_path" in repr_d.keys():
        equilibrium_file_path = repr_d.pop("equilibrium_file_path")
        overridden_variables = []
        for key in key_list:
            if key in repr_d.keys() and repr_d[key] != None :
                overridden_variables.append(key)
        if overridden_variables and equilibrium_file_path:    
            error_message = ("Cannot specify both equilibrium_file_path and:\n{}.\nModify input.yaml so that EITHER the above variables OR equilibrium_file_path is specified."
                         ).format(',\n'.join(overridden_variables))
            raise Exception(error_message)
    else:
        equilibrium_file_path = None

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

    if equilibrium_file_path != None:
        with open(equilibrium_file_path, "r") as f:
            geqdsk = geqdsk.read(f)
        eq_input = {}

        # TO DO: Filling up algorithms that calculate the following variables from the geqdsk file:
        # minor_radius;
        # Plasma_cross-sectional_area
        # seperatrix_elongation
        # 1D array of psi_norm
        # 1D array of triangularity

        value_list = [
        # "major_radius" =
        geqdsk['rmagx'],
        # "magnetic_field_on_axis" = 
        abs(geqdsk['bcentr']),   
        # "inverse_aspect_ratio" = minor_radius/(geqdsk['rmagx'])**2,
        # "areal_elongation" = Plasma_cross-sectional_area/(np.pi * minor_radius**2),
        # "elongation_ratio_sep_to_areal" = seperatrix_elongation/"areal_elongation",
        # "triangularity_psi95" = scipy.interpolate.interp1d(psi_norm,triangularity)(0.95).item(),
        # "triangularity_ratio_sep_to_psi95" = triangularity[-1]/"triangularity_psi95",
        # "plasma_current" = 
        abs(geqdsk['cpasma']),
        ]
        for i in range(len(key_list)):
            eq_input[key_list[i]] = convert_named_options(key=key_list[i], val=value_list[i])

    for key, val in repr_d.items():
        if isinstance(val, (list, tuple)):
            repr_d[key] = [convert_named_options(key=key, val=v) for v in val]
        else:
            repr_d[key] = convert_named_options(key=key, val=val)

    try:
        repr_d = {**repr_d, **eq_input}
    except:
        pass

    for key, val in repr_d.items():
        repr_d[key] = set_default_units(key=key, value=val)

    return repr_d, algorithm, points
