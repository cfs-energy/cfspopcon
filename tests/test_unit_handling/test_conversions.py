import numpy as np

from cfspopcon.unit_handling import Quantity, convert_to_default_units, dimensionless_magnitude, ureg


def test_conversion_of_dimensionless():
    val = Quantity(2.0, ureg.percent)

    assert np.isclose(dimensionless_magnitude(val), 2e-2)
    assert np.isclose(convert_to_default_units(val, key="beta"), 2e-2)
