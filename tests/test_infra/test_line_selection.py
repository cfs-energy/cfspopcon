import numpy as np
import xarray as xr

from cfspopcon.shaping_and_selection.line_selection import find_coords_of_contour, interpolate_onto_line
from cfspopcon.unit_handling import Quantity, convert_units, magnitude, ureg


def test_extract_values_along_contour():
    x_vals = np.linspace(-5, 5, num=500)
    y_vals = np.linspace(-4, 4, num=400)

    x_coord = "dim_x"
    y_coord = "dim_y"

    units = ureg.kW
    level = Quantity(1200.0, ureg.W)

    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_vals = xr.DataArray(np.sqrt(x_mesh**2 + y_mesh**2), coords={y_coord: y_vals, x_coord: x_vals}).pint.quantify(units)

    contour_x, contour_y = find_coords_of_contour(z_vals, x_coord=x_coord, y_coord=y_coord, level=level)

    assert np.allclose(np.sqrt(contour_x**2 + contour_y**2), level.magnitude / 1e3, rtol=1e-3)

    assert np.allclose(
        magnitude(convert_units(interpolate_onto_line(z_vals, contour_x, contour_y), units)),
        magnitude(convert_units(level, units)),
        rtol=1e-3,
    )
