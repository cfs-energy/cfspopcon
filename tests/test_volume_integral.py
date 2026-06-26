import numpy as np
import pytest
import xarray as xr

from cfspopcon.formulas.geometry import integrate_profile_over_volume
from cfspopcon.formulas.plasma_profiles import build_rho_grid
from cfspopcon.unit_handling import magnitude_in_units, ureg


@pytest.mark.parametrize(
    "rho",
    [
        pytest.param(np.array([0.0, 0.2, 0.4, 0.95, 0.99]), id="nonuniform"),
        pytest.param(np.linspace(0.0, 1.0, num=50), id="uniform"),
    ],
)
def test_integrate_profile_over_volume_constant_profile_is_exact(rho):
    """A constant profile must integrate to rho_max**2 * plasma_volume, exactly.

    For f(rho) = 1, the volume integral is int_0^rho_max 1 * 2 * rho drho * V =
    (rho[-1]**2 - rho[0]**2) * V. The trapezoid rule reproduces this exactly,
    because the integrand 2 * rho is linear in rho, independent of the (possibly
    nonuniform) knot placement.
    """
    profile = np.ones_like(rho)
    plasma_volume = 10.0
    expected = rho[-1] ** 2 * plasma_volume

    np.testing.assert_allclose(
        integrate_profile_over_volume.unitless_func(profile, rho, plasma_volume),
        expected,
    )


def test_integrate_profile_over_volume_recovers_plasma_volume_on_production_grid():
    """For a constant profile, the volume integral must recover plasma_volume."""
    rho = build_rho_grid(npoints=50)
    profile = np.ones_like(rho)
    plasma_volume = 10.0

    np.testing.assert_allclose(
        integrate_profile_over_volume.unitless_func(profile, rho, plasma_volume),
        plasma_volume,
        rtol=1e-2,
    )


def test_integrate_profile_over_volume_handles_units_on_irregular_grid():
    """The unit-wrapped entry point matches the hand-calculated exact result."""
    rho = np.array([0.0, 0.2, 0.4, 0.95, 0.99])
    profile = np.ones_like(rho)
    plasma_volume = 10.0
    expected = rho[-1] ** 2 * plasma_volume

    integrated_profile = integrate_profile_over_volume(
        xr.DataArray(profile, dims=["dim_rho"]) / ureg.m**3,
        xr.DataArray(rho, dims=["dim_rho"]) * ureg.dimensionless,
        plasma_volume * ureg.m**3,
    )
    np.testing.assert_allclose(magnitude_in_units(integrated_profile, ureg.dimensionless), expected)
