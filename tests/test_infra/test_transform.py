import pytest
import xarray as xr

from cfspopcon.shaping_and_selection import transform_coords


def test_interpolate_onto_new_coords(z, z1, z2):
    z_interp = transform_coords.interpolate_array_onto_new_coords(array=z, new_coords=dict(z1=z1, z2=z2), default_resolution=5)

    assert z_interp.min() >= z.min()
    assert z_interp.max() <= z.max()
    assert "z1" in z_interp.dims
    assert "z2" in z_interp.dims


def test_order_dimensions(z):
    assert transform_coords.order_dimensions(z, dims=("x", "y"), order_for_plotting=True).dims == ("y", "x")
    assert transform_coords.order_dimensions(z, dims=("x", "y"), order_for_plotting=False).dims == ("x", "y")

    with pytest.raises(ValueError):
        assert transform_coords.order_dimensions(z.isel(x=0), dims=("x", "y")).dims == ("x", "y")

    assert transform_coords.order_dimensions(z.isel(x=0), dims=("x", "y"), template=z).dims == ("y", "x")
    ds = xr.Dataset(dict(z=z))
    assert transform_coords.order_dimensions(z.isel(x=0), dims=("x", "y"), template=ds).dims == ("y", "x")
