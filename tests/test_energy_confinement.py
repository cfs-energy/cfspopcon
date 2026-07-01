import numpy as np
import pytest
import xarray as xr

from cfspopcon.formulas.energy_confinement import (
    calc_energy_confinement_time_from_scaling,
    calc_energy_confinement_time_from_stored_energy_and_input_power,
    calc_H98y2,
    calc_power_balance_from_input_P_aux,
)
from cfspopcon.formulas.energy_confinement.solve_for_input_power import solve_energy_confinement_scaling_for_input_power
from cfspopcon.unit_handling import magnitude_in_units, ureg


@pytest.fixture()
def kwargs():
    return dict(
        confinement_time_scalar=0.90,
        plasma_current=8.7 * ureg.MA,
        magnetic_field_on_axis=12.16 * ureg.T,
        average_electron_density=25.0 * ureg.n19,
        major_radius=1.85 * ureg.m,
        areal_elongation=1.75,
        separatrix_elongation=1.96,
        inverse_aspect_ratio=0.308,
        average_ion_mass=2.5 * ureg.amu,
        triangularity_psi95=0.3,
        separatrix_triangularity=0.54,
        q_star=3.29,
        energy_confinement_scaling="ITER98y2",
    )


def test_tau_e_forward_vs_inverse_calculation(kwargs):

    P_in = 140.0 * ureg.MW

    tau_e = calc_energy_confinement_time_from_scaling(**kwargs, P_in=P_in)
    plasma_stored_energy = P_in * tau_e

    assert tau_e > 0.0 * ureg.seconds
    assert plasma_stored_energy > 0.0 * ureg.MJ

    tau_e_2, input_power_2 = solve_energy_confinement_scaling_for_input_power(**kwargs, plasma_stored_energy=plasma_stored_energy)

    assert np.isclose(magnitude_in_units(tau_e, ureg.s), magnitude_in_units(tau_e_2, ureg.s))
    assert np.isclose(magnitude_in_units(P_in, ureg.MW), magnitude_in_units(input_power_2, ureg.MW))

    tau_e_3 = calc_energy_confinement_time_from_stored_energy_and_input_power(plasma_stored_energy=plasma_stored_energy, P_in=P_in)
    assert np.isclose(magnitude_in_units(tau_e, ureg.s), magnitude_in_units(tau_e_3, ureg.s))


def test_tau_e_inverse_vs_forward_calculation(kwargs):

    plasma_stored_energy = 14.0 * ureg.MJ

    tau_e, P_in = solve_energy_confinement_scaling_for_input_power(**kwargs, plasma_stored_energy=plasma_stored_energy)

    tau_e_2 = calc_energy_confinement_time_from_scaling(**kwargs, P_in=P_in)
    plasma_stored_energy_2 = P_in * tau_e_2

    assert tau_e > 0.0 * ureg.seconds
    assert P_in > 0.0 * ureg.MW

    assert np.isclose(magnitude_in_units(tau_e, ureg.s), magnitude_in_units(tau_e_2, ureg.s))
    assert np.isclose(magnitude_in_units(plasma_stored_energy, ureg.MJ), magnitude_in_units(plasma_stored_energy_2, ureg.MJ))

    tau_e_3 = calc_energy_confinement_time_from_stored_energy_and_input_power(plasma_stored_energy=plasma_stored_energy, P_in=P_in)
    assert np.isclose(magnitude_in_units(tau_e, ureg.s), magnitude_in_units(tau_e_3, ureg.s))


def test_solve_energy_confinement_scaling_reports_required_h98(kwargs):

    test_val = 1.35
    plasma_stored_energy = 14.0 * ureg.MJ

    kwargs_copy = kwargs.copy()
    del kwargs_copy["confinement_time_scalar"]
    del kwargs_copy["energy_confinement_scaling"]

    energy_confinement_time, P_in = solve_energy_confinement_scaling_for_input_power(
        **kwargs_copy, confinement_time_scalar=test_val, plasma_stored_energy=plasma_stored_energy, energy_confinement_scaling="ITER98y2"
    )

    H98y2 = calc_H98y2(
        **kwargs_copy,
        energy_confinement_time=energy_confinement_time,
        P_in=P_in,
    )

    assert magnitude_in_units(energy_confinement_time, ureg.s) > 0.0
    assert magnitude_in_units(P_in, ureg.MW) > 0.0
    np.testing.assert_allclose(magnitude_in_units(H98y2, ureg.dimensionless), 1.35, rtol=1e-6)


def test_calc_power_balance_from_input_p_aux_uses_explicit_alpha_power():
    ds = xr.Dataset(
        data_vars=dict(
            plasma_stored_energy=20.0 * ureg.MJ,
            P_ohmic=1.5 * ureg.MW,
            P_alpha=31.0 * ureg.MW,
            average_electron_density=25.0 * ureg.n19,
            plasma_current=8.7 * ureg.MA,
            major_radius=1.85 * ureg.m,
            inverse_aspect_ratio=0.308,
            magnetic_field_on_axis=12.16 * ureg.T,
            average_ion_mass=2.5 * ureg.amu,
            areal_elongation=1.75,
            separatrix_elongation=1.96,
            triangularity_psi95=0.3,
            separatrix_triangularity=0.54,
            q_star=3.29,
            P_auxiliary_launched=28.0 * ureg.MW,
            fraction_of_external_power_coupled=0.8,
        )
    )

    ds = calc_power_balance_from_input_P_aux.update_dataset(ds)

    np.testing.assert_allclose(magnitude_in_units(ds["P_auxiliary_absorbed"], ureg.MW), 22.4, rtol=1e-9)
    np.testing.assert_allclose(magnitude_in_units(ds["P_in"], ureg.MW), 54.9, rtol=1e-9)
    np.testing.assert_allclose(magnitude_in_units(ds["energy_confinement_time"], ureg.s), 20.0 / 54.9, rtol=1e-9)
    assert magnitude_in_units(ds["H98y2"], ureg.dimensionless) > 0.0
