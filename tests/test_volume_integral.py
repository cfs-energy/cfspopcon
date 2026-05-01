import numpy as np
import xarray as xr

from cfspopcon.formulas.geometry import integrate_profile_over_volume
from cfspopcon.unit_handling import magnitude_in_units, ureg


def test_integrate_profile_over_volume_handles_irregular_rho_grid():
    rho = np.array([0.0, 0.2, 0.4, 0.95, 0.99])
    profile = np.ones_like(rho)
    plasma_volume = 10.0

    expected = np.trapezoid(profile * 2.0 * rho, x=rho) * plasma_volume

    np.testing.assert_allclose(integrate_profile_over_volume.unitless_func(profile, rho, plasma_volume), expected)

    integrated_profile = integrate_profile_over_volume(
        xr.DataArray(profile, dims=["dim_rho"]) / ureg.m**3,
        xr.DataArray(rho, dims=["dim_rho"]) * ureg.dimensionless,
        plasma_volume * ureg.m**3,
    )
    np.testing.assert_allclose(magnitude_in_units(integrated_profile, ureg.dimensionless), expected)
