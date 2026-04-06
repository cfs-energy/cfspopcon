import numpy as np

from cfspopcon.formulas.plasma_profiles import calc_1D_plasma_profiles
from cfspopcon.named_options import ProfileForm
from cfspopcon.unit_handling import magnitude_in_units, ureg


def test_calc_1d_plasma_profiles_jch_respects_requested_peaking():
    rho, electron_density, fuel_ion_density, electron_temp, ion_temp = calc_1D_plasma_profiles(
        density_profile_form=ProfileForm.jch,
        temp_profile_form=ProfileForm.jch,
        average_electron_density=20.0 * ureg.n19,
        average_electron_temp=10.0 * ureg.keV,
        average_ion_temp=12.0 * ureg.keV,
        electron_density_peaking=1.5,
        ion_density_peaking=1.2,
        temperature_peaking=2.0,
        dilution=0.8,
        normalized_inverse_temp_scale_length=2.5,
        pedestal_width=0.05,
        t_sep=0.2 * ureg.keV,
        n_sep_ratio=0.5,
    )

    rho_mag = magnitude_in_units(rho, ureg.dimensionless)
    electron_density_mag = magnitude_in_units(electron_density, ureg.n19)
    fuel_ion_density_mag = magnitude_in_units(fuel_ion_density, ureg.n19)
    electron_temp_mag = magnitude_in_units(electron_temp, ureg.keV)
    ion_temp_mag = magnitude_in_units(ion_temp, ureg.keV)

    assert rho_mag[0] == 0.0
    assert rho_mag[-1] < 1.0

    np.testing.assert_allclose(electron_density_mag[0], 30.0, rtol=1e-4)
    np.testing.assert_allclose(fuel_ion_density_mag[0], 19.2, rtol=1e-4)
    np.testing.assert_allclose(electron_temp_mag[0], 20.0, rtol=1e-4)
    np.testing.assert_allclose(ion_temp_mag[0], 24.0, rtol=1e-4)
