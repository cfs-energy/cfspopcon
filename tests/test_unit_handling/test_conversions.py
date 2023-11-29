import warnings

import numpy as np
import pytest
import xarray as xr

from cfspopcon.unit_handling import dimensionless_magnitude, ureg, Quantity, ureg, convert_to_default_units


def test_conversion_of_dimensionless():

    val = Quantity(2.0, ureg.percent)

    assert np.isclose(dimensionless_magnitude(val), 2e-2)
    assert np.isclose(convert_to_default_units(val, key="beta"), 2e-2)
