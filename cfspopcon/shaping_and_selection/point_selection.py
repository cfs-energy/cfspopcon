"""Routines to find the coordinates of the minimum or maximum value of a field."""

import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates

from ..unit_handling import Quantity, dimensionless_magnitude, magnitude_in_default_units


def find_values_at_nearest_point(dataset: xr.Dataset, point_params: dict) -> xr.Dataset:
    """Return a dataset at a point point which best fulfills the conditions defined by point params."""
    allowed_methods = ["minimize", "maximize", "nearest_to", "interp_to"]

    method = [method for method in allowed_methods if method in point_params.keys()]
    assert len(method) == 1, f"Must provide one of [{', '.join(allowed_methods)}] for a point. Keys were {list(point_params.keys())}"

    if method[0] == "interp_to":
        mask = build_mask_from_dict(dataset, point_params)

        requested_coords = dict()

        for dimension_name, request in point_params["interp_to"].items():
            if dimension_name not in dataset.coords and f"dim_{dimension_name}" in dataset.coords:
                dimension_name = f"dim_{dimension_name}"  # noqa: PLW2901

            assert dimension_name in dataset.coords, (
                f"Cannot interpolate to {dimension_name} since it is not in the dataset coordinates {dataset.coords}."
            )

            value = Quantity(float(request["value"]), request.get("units", ""))
            requested_coords[dimension_name] = magnitude_in_default_units(value, key=dimension_name.lstrip("dim_"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We dequantify and then requantify the dataset, since some interpolation methods cannot handle units
            return (  # type:ignore[no-any-return]
                dataset.where(mask)
                .pint.dequantify()
                .interp(**requested_coords, method=point_params.get("method", "linear"))
                .pint.quantify()
            )

    elif method[0] in ["minimize", "maximize", "nearest_to"]:
        return dataset.isel(find_coords_of_nearest_point(dataset, point_params))
    else:
        raise NotImplementedError(f"{method[0]} not recognized.")


def find_coords_of_nearest_point(dataset: xr.Dataset, point_params: dict) -> DataArrayCoordinates:
    """Find the coordinates of a point which best fulfills the conditions defined by point params.

    The point parameters must have a 'minimize', 'maximize' or 'nearest_to' key.

    'point_name': {
        'maximize': 'Q'
    }
    will find the point with the highest 'Q' value

    'point_name': {
        'nearest_to': {
            'average_electron_density': {
                'value': 20.0,
                'units': 'n19'
            },
            'max_flattop_duration': {
                'value': 2.0,
                'norm': 1.0,
                'units': 's'
            },
        },
        'tolerance': 1e-2,
    }
    will find the point which minimizes
    d = sqrt(
        ((average_electron_density - (20 * n19)) / (20 * n19))**2
      + ((max_flattop_duration - (2 * s)) / (1 * s))**2
    )
    If the resulting point has d > 1e-2 (tolerance), an AssertionError will be raised.

    'nearest_to' is intended to return the dataset at a given grid point. However, you
    can use it to find points fulfilling non-scanned conditions (as in the example above).

    A mask can also be provided.
    'point_name': {
        'maximize': 'Q',
        'where': {
            'P_auxiliary_launched': {
                'min': 0.0,
                'max': 25.0,
                'units': 'MW',
            },
            'greenwald_fraction': {
                'max': 0.9
            }
        }
    }
    will find the maximum value of Q in the region with P_auxiliary_launched between 0 and 25MW, and with a Greenwald
    fraction up to 90%.
    """
    allowed_methods = ["minimize", "maximize", "nearest_to"]

    method = [method for method in allowed_methods if method in point_params.keys()]
    assert len(method) == 1, f"Must provide one of [{', '.join(allowed_methods)}] for a point. Keys were {list(point_params.keys())}"

    mask = build_mask_from_dict(dataset, point_params)

    if method[0] == "minimize":
        coords_of_point = find_coords_of_minimum(dataset[point_params["minimize"]], keep_dims=point_params.get("keep_dims", []), mask=mask)
    elif method[0] == "maximize":
        coords_of_point = find_coords_of_maximum(dataset[point_params["maximize"]], keep_dims=point_params.get("keep_dims", []), mask=mask)
    elif method[0] == "nearest_to":
        normalized_distance_squared = []
        for variable_name, request in point_params["nearest_to"].items():
            value = Quantity(float(request["value"]), request.get("units", ""))
            normalization = Quantity(float(request.get("norm", request["value"])), request.get("units", ""))
            normalized_distance_squared.append(dimensionless_magnitude(((dataset[variable_name] - value) / normalization) ** 2))

        euclidean_distance = np.sqrt(xr.concat(xr.broadcast(*normalized_distance_squared), dim="dim_distance").sum(dim="dim_distance"))  # type:ignore[call-overload]
        coords_of_point = find_coords_of_minimum(euclidean_distance, keep_dims=point_params.get("keep_dims", []), mask=mask)

        if "tolerance" in point_params:
            assert np.all((tol := euclidean_distance.isel(coords_of_point)) < point_params["tolerance"]), (
                f"Normalized distance at nearest point [{tol.values}] is greater than the requested tolerance [{point_params['tolerance']}]"
            )
    else:
        raise NotImplementedError(f"{method[0]} not recognized.")

    return coords_of_point


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
