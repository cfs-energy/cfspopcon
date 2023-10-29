import numpy as np
import pytest
import xarray as xr


@pytest.fixture()
def x():
    return np.linspace(0, 10, num=400)


@pytest.fixture()
def y():
    return np.linspace(-5, 5, num=500)


@pytest.fixture()
def z(x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    return xr.DataArray(x_grid + y_grid, coords=dict(y=y, x=x))


@pytest.fixture()
def z1(x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    return xr.DataArray(x_grid + y_grid**2, coords=dict(y=y, x=x))


@pytest.fixture()
def z2(x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    return xr.DataArray(x_grid**2 + y_grid, coords=dict(y=y, x=x))


@pytest.fixture()
def z3(x, y):
    x_grid, y_grid = np.meshgrid(x, y)
    return xr.DataArray(np.abs(y_grid + x_grid), coords=dict(y=y, x=x))


@pytest.fixture()
def ds(x, y, z, z1, z2, z3):
    return xr.Dataset(dict(x=x, y=y, z=z, z1=z1, z2=z2, z3=z3))
