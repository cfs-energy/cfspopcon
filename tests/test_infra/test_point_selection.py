import numpy as np

from cfspopcon.point_selection import find_coords_of_maximum


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
