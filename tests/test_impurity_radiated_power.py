"""Physical species, units, and density scaling of the analytical radiation fits."""

import numpy as np
import pytest
import xarray as xr

from cfspopcon.formulas.radiated_power import (
    calc_impurity_radiated_power_mavrin_coronal,
    calc_impurity_radiated_power_mavrin_noncoronal,
    calc_impurity_radiated_power_post_and_jensen,
)
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg


@pytest.fixture()
def rho():
    """A radial profile with the dimension required by the public API."""
    return xr.DataArray(np.linspace(0.0, 1.0, 101), dims="dim_rho") * ureg.dimensionless


@pytest.mark.parametrize(
    ("model", "species", "temp_keV", "cooling_rate", "residence_time"),
    [
        (calc_impurity_radiated_power_mavrin_coronal, AtomicSpecies.Helium, 10.0, 6.259505134393894e-36, None),
        (calc_impurity_radiated_power_mavrin_coronal, AtomicSpecies.Tungsten, 10.0, 1.312230113886328e-31, None),
        (calc_impurity_radiated_power_mavrin_coronal, AtomicSpecies.Helium, 100.0, 1.364897379644951e-35, None),
        (calc_impurity_radiated_power_post_and_jensen, AtomicSpecies.Helium, 10.0, 7.502957734521961e-36, None),
        (calc_impurity_radiated_power_post_and_jensen, AtomicSpecies.Tungsten, 10.0, 1.192638980974382e-31, None),
        (calc_impurity_radiated_power_post_and_jensen, AtomicSpecies.Helium, 100.0, 2.228106779126558e-35, None),
        (calc_impurity_radiated_power_mavrin_noncoronal, AtomicSpecies.Helium, 10.0, 7.215725442590215e-36, 0.1),
        (calc_impurity_radiated_power_mavrin_noncoronal, AtomicSpecies.Helium, 0.01, 4.073990387911941e-34, 0.001),
    ],
)
def test_uniform_profile_radiation(rho, model, species, temp_keV, cooling_rate, residence_time):
    """For uniform profiles, P = Lz * ne**2 * concentration * volume.

    Cooling rates in W m^3 were evaluated separately from the existing fit
    coefficients, not obtained from the function under test. At 10 keV the
    coronal helium log10(Lz) is -35.20346; at 100 keV it is -34.86490.
    Post & Jensen's cgs rates include the 1e-13 conversion to SI. The
    noncoronal cases use X=4,Y=0 and X=1,Y=-2 respectively, with
    X=log10(Te[eV]) and Y=log10(ne[m^-3]*tau[s]/1e19).
    """
    kwargs = {}
    if residence_time is not None:
        kwargs["impurity_residence_time"] = residence_time * ureg.s
    power = model(
        rho=rho,
        electron_temp_profile=xr.full_like(rho, temp_keV) * ureg.keV,
        electron_density_profile=xr.full_like(rho, 10.0) * ureg.n19,
        impurity_concentration=0.01 * ureg.dimensionless,
        impurity_species=species,
        plasma_volume=100.0 * ureg.m**3,
        **kwargs,
    )
    expected_MW = cooling_rate * (1e20) ** 2 * 0.01 * 100.0 / 1e6
    np.testing.assert_allclose(magnitude_in_units(power, ureg.MW), expected_MW, rtol=1e-10)


def test_mavrin_coronal_density_scaling(rho):
    """Changing density must not change the temperature fit or clip density."""
    # Both values exceed 100 in n19 units: the old code clipped them as if keV.
    density = xr.DataArray([150.0, 300.0], dims="dim_density") * ureg.n19
    power = calc_impurity_radiated_power_mavrin_coronal(
        rho=rho,
        electron_temp_profile=xr.full_like(rho, 10.0) * ureg.keV,
        electron_density_profile=xr.ones_like(rho) * density,
        impurity_concentration=0.01 * ureg.dimensionless,
        impurity_species=AtomicSpecies.Helium,
        plasma_volume=100.0 * ureg.m**3,
    )
    values = np.asarray(magnitude_in_units(power, ureg.MW))
    np.testing.assert_allclose(values[1] / values[0], 4.0, rtol=1e-12)


def test_unsupported_species_do_not_select_another_element(rho):
    """An enum ordinal must not select a different element's supported fit."""
    with pytest.warns(UserWarning, match="not supported"):
        power = calc_impurity_radiated_power_mavrin_coronal(
            rho=rho,
            electron_temp_profile=xr.full_like(rho, 10.0) * ureg.keV,
            electron_density_profile=xr.full_like(rho, 10.0) * ureg.n19,
            impurity_concentration=0.01 * ureg.dimensionless,
            impurity_species=AtomicSpecies.Boron,
            plasma_volume=100.0 * ureg.m**3,
        )
    assert np.isnan(magnitude_in_units(power, ureg.MW))
