"""Functions for saving results to file and loading those files."""
import json
from pathlib import Path
from typing import Any

import xarray as xr
import numpy as np

from .helpers import convert_named_options
from .point_selection import build_mask_from_dict, find_coords_of_minimum
from .unit_handling import convert_to_default_units, set_default_units


def sanitize_variable(val: xr.DataArray, key: str) -> xr.DataArray:
    """Strip units and Enum values from a variable so that it can be stored in a NetCDF file."""
    try:
        val = convert_to_default_units(val, key).pint.dequantify()
    except KeyError:
        pass

    if val.dtype == object:
        if val.size == 1:
            val = val.item().name
        else:
            val = xr.DataArray([v.name for v in val.values])

    return val


def write_dataset_to_netcdf(dataset: xr.Dataset, filepath: Path) -> None:
    """Write a dataset to a NetCDF file."""
    serialized_dataset = dataset.copy()
    for key in dataset.keys():
        assert isinstance(key, str)  # because hashable type of key is broader str but we  know it's str
        serialized_dataset[key] = sanitize_variable(dataset[key], key)

    for key in dataset.coords:
        assert isinstance(key, str)  # because hashable type of key is broader str but we  know it's str
        serialized_dataset[key] = sanitize_variable(dataset[key], key)

    serialized_dataset.to_netcdf(filepath)


def promote_variable(val: xr.DataArray, key: str) -> Any:
    """Add back in units and Enum values that were removed by sanitize_variable."""
    try:
        val = set_default_units(val, key)
    except KeyError:
        pass

    if val.dtype == object or val.dtype.type == np.str_:
        if val.size == 1:
            return convert_named_options(key, val.item())
        else:
            return [convert_named_options(key, v) for v in val.values]

    return val


def read_dataset_from_netcdf(filepath: Path) -> xr.Dataset:
    """Open a dataset from a NetCDF file."""
    dataset = xr.open_dataset(filepath)

    for key in dataset.keys():
        assert isinstance(key, str)
        dataset[key] = promote_variable(dataset[key], key)

    for key in dataset.coords:
        if key == "dim_species":
            dataset[key] = promote_variable(dataset[key], key="impurity")
        else:
            assert isinstance(key, str)  # because hashable type of key is broader str but we  know it's str
            dataset[key] = promote_variable(dataset[key], key)

    return dataset


def write_point_to_file(dataset: xr.Dataset, point_key: str, point_params: dict, output_dir: Path) -> None:
    """Write the analysis values at the named points to a json file."""
    mask = build_mask_from_dict(dataset, point_params)

    if "minimize" not in point_params.keys() and "maximize" not in point_params.keys():
        raise ValueError(f"Need to provide either minimize or maximize in point specification. Keys were {point_params.keys()}")

    if "minimize" in point_params.keys():
        array = dataset[point_params["minimize"]]
    else:
        array = -dataset[point_params["maximize"]]

    point_coords = find_coords_of_minimum(array, keep_dims=point_params.get("keep_dims", []), mask=mask)

    point = dataset.isel(point_coords)

    for key in point.keys():
        assert isinstance(key, str)  # because hashable type of key is broader str but we  know it's str
        point[key] = sanitize_variable(point[key], key)

    for key in point.coords:
        assert isinstance(key, str)  # because hashable type of key is broader str but we  know it's str
        point[key] = sanitize_variable(point[key], key)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{point_key}.json", "w") as file:
        json.dump(point.to_dict(), file, indent=4)
