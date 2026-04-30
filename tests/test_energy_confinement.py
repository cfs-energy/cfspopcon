import numpy as np

from cfspopcon.formulas.energy_confinement import calc_power_balance_from_input_P_aux
from cfspopcon.formulas.energy_confinement.read_energy_confinement_scalings import read_confinement_scalings
from cfspopcon.formulas.energy_confinement.solve_for_input_power import solve_energy_confinement_scaling_for_input_power
from cfspopcon.unit_handling import magnitude_in_units, ureg


def test_solve_energy_confinement_scaling_reports_required_h98():
    read_confinement_scalings()

    tau_e, p_in, required_h98 = solve_energy_confinement_scaling_for_input_power(
        confinement_time_scalar=1.35,
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
        plasma_stored_energy=20.0 * ureg.MJ,
        q_star=3.29,
        energy_confinement_scaling="ITER98y2",
    )

    assert magnitude_in_units(tau_e, ureg.s) > 0.0
    assert magnitude_in_units(p_in, ureg.MW) > 0.0
    np.testing.assert_allclose(magnitude_in_units(required_h98, ureg.dimensionless), 1.35, rtol=1e-6)


def test_calc_power_balance_from_input_p_aux_uses_explicit_alpha_power():
    p_in, p_aux_absorbed, tau_e, required_h98 = calc_power_balance_from_input_P_aux(
        plasma_stored_energy=20.0 * ureg.MJ,
        P_ohmic=1.5 * ureg.MW,
        P_alpha=31.0 * ureg.MW,
        average_electron_density=25.0 * ureg.n19,
        plasma_current=8.7 * ureg.MA,
        major_radius=1.85 * ureg.m,
        minor_radius=0.57 * ureg.m,
        magnetic_field_on_axis=12.16 * ureg.T,
        average_ion_mass=2.5 * ureg.amu,
        areal_elongation=1.75,
        P_auxiliary_launched=28.0 * ureg.MW,
        fraction_of_external_power_coupled=0.8,
    )

    np.testing.assert_allclose(magnitude_in_units(p_aux_absorbed, ureg.MW), 22.4, rtol=1e-9)
    np.testing.assert_allclose(magnitude_in_units(p_in, ureg.MW), 54.9, rtol=1e-9)
    np.testing.assert_allclose(magnitude_in_units(tau_e, ureg.s), 20.0 / 54.9, rtol=1e-9)
    assert magnitude_in_units(required_h98, ureg.dimensionless) > 0.0
