"""Routines to find the coordinates of the minimum or maximum value of a field."""
from collections.abc import Sequence
from typing import Optional

import numpy as np
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates

from ..unit_handling import Quantity


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
