"""Functions for saving results to file and loading those files."""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, Union

if sys.version_info >= (3, 11, 0):
    from typing import Self  # type:ignore[attr-defined,unused-ignore]
else:
    from typing_extensions import Self  # type:ignore[attr-defined,unused-ignore]

import numpy as np
import xarray as xr

from .helpers import convert_named_options
from .shaping_and_selection.point_selection import build_mask_from_dict, find_coords_of_minimum
from .unit_handling import convert_to_default_units, set_default_units

ignored_keys = [
    "radas_dir",
    "atomic_data",
]


def sanitize_variable(val: xr.DataArray, key: str, coord: bool = False) -> Union[xr.DataArray, str]:
    """Strip units and Enum values from a variable so that it can be stored in a NetCDF file.

    If you set coord=True and you pass in a scalar val, val is wrapped in a length-1 array to
    circumvent an xarray issue regarding single-value coordinates.
    See https://github.com/pydata/xarray/issues/1709.
    """
    try:
        val = convert_to_default_units(val, key).pint.dequantify()
    except KeyError:
        pass

    if val.dtype == object:
        try:
            if val.size == 1:
                if not coord:
                    val = val.item().name
                else:
                    val = xr.DataArray([val.item().name])
            else:
                val = xr.DataArray([v.name for v in val.values], dims=val.dims)
        except AttributeError:
            warnings.warn(f"Cannot handle {key}. Dropping variable.", stacklevel=3)
            return "UNHANDLED"

    return val


def write_dataset_to_netcdf(
    dataset: xr.Dataset, filepath: Path, netcdf_writer: Literal["netcdf4", "scipy", "h5netcdf"] = "netcdf4"
) -> None:
    """Write a dataset to a NetCDF file."""
    serialized_dataset = dataset.copy()
    for key in ignored_keys:
        assert isinstance(key, str)  # for type-checking
        # errors="ignore" prevents drop_vars from raising a ValueError if key
        # is not in serialized_dataset. This is necessary for writing a dataset
        # that has previously been read from file (i.e. round-trip file I/O)
        serialized_dataset = serialized_dataset.drop_vars(key, errors="ignore")

    for key in serialized_dataset.keys():  # type:ignore[assignment]
        assert isinstance(key, str)
        serialized_dataset[key] = sanitize_variable(dataset[key], key)

    for key in serialized_dataset.coords:  # type:ignore[assignment]
        assert isinstance(key, str)
        serialized_dataset[key] = sanitize_variable(dataset[key], key, coord=True)

    serialized_dataset.to_netcdf(filepath, engine=netcdf_writer)


def promote_variable(val: xr.DataArray, key: str) -> Any:
    """Add back in units and Enum values that were removed by sanitize_variable."""
    try:
        val = set_default_units(val, key)
    except KeyError:
        pass

    # scipy i/o yields objects for strings while netcdf4 yields np.str_
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
            assert isinstance(key, str)
            dataset[key] = promote_variable(dataset[key], key)

    return dataset


# These following lines are needed to modify the representation of floats
# in the JSON file. This is needed, because otherwise small errors in the
# floats lead to a large, meaningless diff of the reference JSON files.


class _RoundingFloat(float):
    """A modified version of the 'float' built-in, with a modified __repr__.

    This is needed because `iterencode` directly uses `float.__repr__` in `floatstr`.

    From: https://stackoverflow.com/questions/54370322/how-to-limit-the-number-of-float-digits-jsonencoder-produces
    """

    __repr__ = staticmethod(lambda x: f"{x:#.10g}")  # type:ignore[assignment,unused-ignore]


class _ModifyJSONFloatRepr:
    """A ContextManager to locally modify the representation of floats."""

    def __enter__(self) -> Self:
        """Change the float representation to fixed precision on entry."""
        self.c_make_encoder = json.encoder.c_make_encoder  # type:ignore[attr-defined]
        json.encoder.c_make_encoder = None  # type:ignore[attr-defined]
        json.encoder.float = _RoundingFloat  # type:ignore[attr-defined]
        return self

    def __exit__(self, *args: Any) -> None:
        """Change the float representation back to the default."""
        json.encoder.c_make_encoder = self.c_make_encoder  # type:ignore[attr-defined]
        json.encoder.float = float  # type:ignore[attr-defined]


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
        if key in ignored_keys:
            assert isinstance(key, str)
            point = point.drop_vars(key, errors="ignore")

    for key in point.keys():
        assert isinstance(key, str)
        point[key] = sanitize_variable(point[key], key)

    for key in point.coords:
        assert isinstance(key, str)
        point[key] = sanitize_variable(point[key], key)

    output_dir.mkdir(parents=True, exist_ok=True)

    with _ModifyJSONFloatRepr():
        with open(output_dir / f"{point_key}.json", "w") as file:
            json.dump(point.to_dict(), file, indent=4, sort_keys=True)
