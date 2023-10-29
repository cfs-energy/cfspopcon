"""Adds a readout of the field at the current mouse position for a colormapped field plotted with pcolormesh, contour, quiver, etc.

Usage:
>>> fig, ax = plt.subplots()
>>> ax.format_coord = CoordinateFormatter(x, y, z)
"""
import xarray as xr


class CoordinateFormatter:
    """Data storage object used for providing a coordinate formatter."""

    def __init__(self, array: xr.DataArray):  # pragma: nocover
        """Stores the data required for grid lookup."""
        self.array = array

    def __call__(self, mouse_x, mouse_y):  # pragma: nocover
        """Returns a string which gives the field value at the queried mouse position."""
        lookup = dict(zip(self.array.dims, (mouse_y, mouse_x)))

        mouse_z = float(self.array.sel(lookup, method="nearest").item())

        return f"x={mouse_x:f}, y={mouse_y:f}, z={mouse_z:f}"
