"""Routines to extract the values of an array along a line of points (typically a contour, but interpolate_onto_line is flexible)."""

from typing import TYPE_CHECKING, cast

import xarray as xr
from contourpy import contour_generator
from scipy.interpolate import RegularGridInterpolator

from cfspopcon.unit_handling import Quantity, convert_units, get_units, magnitude

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def find_coords_of_contour(array: xr.DataArray, x_coord: str, y_coord: str, level: Quantity) -> tuple[xr.DataArray, xr.DataArray]:
    """Find the x and y values of a contour of the input array."""
    units = get_units(array)

    cont_gen = contour_generator(
        x=array.coords[y_coord],
        y=array.coords[x_coord],
        z=magnitude(array.transpose(x_coord, y_coord)),
    )

    lines = cont_gen.lines(magnitude(convert_units(level, units)))
    if not len(lines) == 1:
        raise RuntimeError(
            f"find_coords_of_contour returned {len(lines)} contours at level {level}. Use masks to isolate a single contour."
        )
    # contourpy types its return as a union covering all line types; the default line type gives one array per line
    line = cast("NDArray[np.float64]", lines[0])

    contour_x = xr.DataArray(line[:, 1], coords={x_coord: line[:, 1]})
    contour_y = xr.DataArray(line[:, 0], coords={x_coord: line[:, 1]})
    contour_x.name, contour_y.name = x_coord, y_coord
    return contour_x, contour_y


def interpolate_onto_line(
    array: xr.DataArray, line_x: xr.DataArray, line_y: xr.DataArray, interpolation_method: str = "cubic"
) -> xr.DataArray:
    """Return values of array at the positions given by line_x and line_y.

    line_x and line_y must have names matching the coordinates of array. You can set their names using
    >>> contour_x.name, contour_y.name = x_coord, y_coord
    """
    units = get_units(array)

    if not array.ndim == 2:
        raise RuntimeError("Contour interpolation only supported for 2 dimensions. Use apply_ufunc to apply to higher dimensions.")

    x_coord, y_coord = line_x.name, line_y.name
    array = array.transpose(x_coord, y_coord)

    interpolator = RegularGridInterpolator(
        points=((array.coords[x_coord], array.coords[y_coord])),
        values=magnitude(array).to_numpy(),
        method=interpolation_method,
    )

    interpolated_values = interpolator((line_x, line_y))

    return xr.DataArray(interpolated_values, coords={x_coord: line_x}).pint.quantify(units)  # type:ignore[no-any-return]
