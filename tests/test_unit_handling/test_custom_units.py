import numpy as np

from cfspopcon.unit_handling import Quantity, ureg


def test_custom_units():
    assert np.isclose(Quantity(10.0, ureg.n19), Quantity(1.0, ureg.n20))
    assert np.isclose(Quantity(1.0, ureg.n19), Quantity(1e19, ureg.m**-3))
    assert np.isclose(Quantity(1.0, ureg.n20), Quantity(1e20, ureg.m**-3))

    assert np.isclose(Quantity(100.0, ureg.percent), Quantity(1.0, ureg.dimensionless))
    assert np.isclose(Quantity(100.0, ureg.percent), 1.0)
