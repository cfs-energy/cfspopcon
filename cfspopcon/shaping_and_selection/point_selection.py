"""Routines to find the coordinates of the minimum or maximum value of a field."""

from collections.abc import Sequence
from typing import Callable, Optional

import numpy as np
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates

from ..unit_handling import Quantity, default_unit, magnitude_in_units


def find_point(dataset: xr.Dataset, point_params: dict, transform_func: Optional[Callable[[xr.DataArray], xr.DataArray]] = None) -> xr.Dataset:
    """Extract values from a dataset at a point defined by point params."""
    allowed_methods = ["minimize", "maximize", "at"]

    assert np.sum([method in point_params.keys() for method in allowed_methods]) == 1, \
        f"Must provide exactly one of [{', '.join(allowed_methods)}] for a point. Keys were {point_params.keys()}"
    
    if transform_func is None:
        def transform_func(x):
            return x

    if "minimize" in point_params.keys() or "maximize" in point_params.keys():
        mask = build_mask_from_dict(dataset, point_params)

        if "minimize" in point_params.keys():
            array = dataset[point_params["minimize"]]
        elif "maximize" in point_params.keys():
            array = -dataset[point_params["maximize"]]

        transformed_array = transform_func(array.where(mask))
        point_coords = find_coords_of_minimum(transformed_array, keep_dims=point_params.get("keep_dims", []))

        return dataset.isel(point_coords)

    elif "at" in point_params.keys():

        method = point_params["at"].get("method", "nearest")

        point_coords = dict()
        for key, at_config in point_params["at"].items():
            if f"dim_{key}" not in dataset.coords:
                raise NotImplementedError(f"Points must be defined in terms of scanned coordinates. dim_{key} not dataset coords {dataset.coords}")
            # Coordinates are defined in default units
            coord_units = default_unit(key)
            value = magnitude_in_units(Quantity(float(at_config["value"]), at_config.get("units", "")), coord_units)
            point_coords[f"dim_{key}"] = value

        if method == "nearest":
            return dataset.sel(**point_coords, method="nearest")
        else:
            return dataset.interp(**point_coords, method=method)

    else:
        raise NotImplementedError(f"No method recognized for point. Please provide one of [{', '.join(allowed_methods)}].")


def find_coords_of_minimum(array: xr.DataArray, keep_dims: Sequence[str] = [], mask: Optional[xr.DataArray] = None) -> DataArrayCoordinates:
    """Find the coordinates the minimum value of array.

    These coordinates can be used to find the value of other arrays at the same point.

    For example
        >>> import xarray as xr
        >>> import numpy as np
        >>> from cfspopcon.point_selection import find_coords_of_minimum
        >>> x = xr.DataArray(np.linspace(0, 1, num=10), dims="x")
        >>> y = xr.DataArray(np.linspace(-1, 1, num=20), dims="y")
        >>> z = xr.DataArray(np.abs(x + y), coords=dict(x=x, y=y))
        >>> line = z.isel(find_coords_of_minimum(z, keep_dims="y"))
    """
    large = Quantity(np.inf, array.pint.units)

    if mask is None:
        masked_array = array.fillna(large)
    else:
        masked_array = array.where(mask).fillna(large)

    along_dims = [dim for dim in masked_array.dims if dim not in keep_dims]
    point_coords = masked_array.argmin(dim=along_dims)

    return point_coords  # type:ignore[return-value]


def find_coords_of_maximum(array: xr.DataArray, keep_dims: Sequence[str] = [], mask: Optional[xr.DataArray] = None) -> DataArrayCoordinates:
    """Find the coordinates of the maximum value of array."""
    return find_coords_of_minimum(-array, keep_dims=keep_dims, mask=mask)


def build_mask_from_dict(dataset: xr.Dataset, plot_params: dict) -> xr.DataArray:
    """Build a mask field which hides inaccessible parts of the operational space."""
    mask = xr.DataArray(True)

    for mask_key, mask_range in plot_params.get("where", dict()).items():
        mask_field = dataset[mask_key]
        mask_min = Quantity(mask_range.get("min", -np.inf), mask_range.get("units", ""))
        mask_max = Quantity(mask_range.get("max", +np.inf), mask_range.get("units", ""))

        mask = mask & ((mask_field > mask_min) & (mask_field < mask_max))

    return mask
