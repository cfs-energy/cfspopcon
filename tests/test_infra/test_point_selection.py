import numpy as np
import xarray as xr

from cfspopcon.shaping_and_selection.point_selection import find_coords_of_maximum, find_values_at_nearest_point


def test_find_coords(ds):
    coords = find_coords_of_maximum(ds.z3)
    assert np.isclose(ds.z3.max(), ds.isel(coords).z3)

    coords = find_coords_of_maximum(ds.z3, keep_dims="x")
    assert np.allclose(ds.z3.max(dim="y"), ds.isel(coords).z3)

    coords = find_coords_of_maximum(ds.z3, keep_dims="y")
    assert np.allclose(ds.z3.max(dim="x"), ds.isel(coords).z3)

    mask = (ds.y < 1.0) & (ds.x < 5.0)

    coords = find_coords_of_maximum(ds.z3, mask=mask)
    # z3 is x + y, so must be close to but less than 1 + 5
    assert np.isclose(ds.isel(coords).z3, 6.0, atol=0.1)


def test_interp_to_dimension_starting_with_d_i_or_m():
    # str.lstrip("dim_") strips any of the characters d, i, m and _, so it used to turn
    # "dim_magnetic_field_on_axis" into "agnetic_field_on_axis".
    magnetic_field = np.linspace(10.0, 14.0, num=5)
    dataset = xr.Dataset(
        dict(Q=xr.DataArray(np.arange(5.0), coords=dict(dim_magnetic_field_on_axis=magnetic_field))),
    )

    point = find_values_at_nearest_point(dataset, {"interp_to": {"magnetic_field_on_axis": {"value": 12.0, "units": "T"}}})

    assert np.isclose(point["Q"], 2.0)
