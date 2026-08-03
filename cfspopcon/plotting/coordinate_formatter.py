"""Adds a readout of the field at the current mouse position for a colormapped field plotted with pcolormesh, contour, quiver, etc.

Usage:
    >>> import matplotlib.pyplot as plt
    >>> from cfspopcon.plotting import CoordinateFormatter
    >>> fig, ax = plt.subplots()
    >>> ax.format_coord = CoordinateFormatter(...)
"""

import xarray as xr


class CoordinateFormatter:
    """Data storage object used for providing a coordinate formatter."""

    def __init__(self, array: xr.DataArray):
        """Stores the data required for grid lookup, moving any units into the attrs.

        Without the dequantify, item() on a unitful field returns a pint Quantity, and float() on
        one raises a DimensionalityError on every mouse move.
        """
        self.array = array.pint.dequantify(format="~P")

    def __call__(self, mouse_x, mouse_y):  # pragma: nocover
        """Returns a string which gives the field value at the queried mouse position."""
        lookup = dict(zip(self.array.dims, (mouse_y, mouse_x), strict=False))

        mouse_z = float(self.array.sel(lookup, method="nearest").item())

        readout = f"x={mouse_x:f}, y={mouse_y:f}, z={mouse_z:f}"
        units = self.array.attrs.get("units", "")
        return f"{readout} [{units}]" if units else readout
