import numpy as np

from cfspopcon.formulas.energy_confinement.read_energy_confinement_scalings import read_confinement_scalings
from cfspopcon.formulas.energy_confinement.solve_for_input_power import solve_energy_confinement_scaling_for_input_power
from cfspopcon.formulas.energy_confinement.switch_confinement_scaling_on_threshold import switch_to_L_mode_confinement_below_threshold
from cfspopcon.unit_handling import magnitude_in_units, ureg


def test_switch_to_L_mode_confinement_below_threshold():
    kwargs = dict(
        plasma_stored_energy=20.0 * ureg.MJ,
        average_electron_density=25.0 * ureg.n19,
        confinement_time_scalar=1.0,
        plasma_current=8.7 * ureg.MA,
        magnetic_field_on_axis=12.16 * ureg.T,
        major_radius=1.85 * ureg.m,
        areal_elongation=1.75,
        separatrix_elongation=1.96,
        inverse_aspect_ratio=0.308,
        average_ion_mass=2.5 * ureg.amu,
        triangularity_psi95=0.3,
        separatrix_triangularity=0.54,
        q_star=3.29,
    )

    read_confinement_scalings()

    tau_E_H_mode, P_in_H_mode = solve_energy_confinement_scaling_for_input_power(
        **kwargs,
        energy_confinement_scaling="ITER98y2",
    )

    tau_E_L_mode, _ = solve_energy_confinement_scaling_for_input_power(
        **kwargs,
        energy_confinement_scaling="ITER89P",
    )

    tau_E_should_be_L_mode, _ = switch_to_L_mode_confinement_below_threshold(
        **kwargs,
        ratio_of_P_SOL_to_P_LH=0.90,
        energy_confinement_time=tau_E_H_mode,
        P_in=P_in_H_mode,
        energy_confinement_scaling_for_L_mode="ITER89P",
    )

    tau_E_should_be_H_mode, _ = switch_to_L_mode_confinement_below_threshold(
        **kwargs,
        ratio_of_P_SOL_to_P_LH=1.2,
        energy_confinement_time=tau_E_H_mode,
        P_in=P_in_H_mode,
        energy_confinement_scaling_for_L_mode="ITER89P",
    )

    np.testing.assert_allclose(magnitude_in_units(tau_E_should_be_L_mode, ureg.s), magnitude_in_units(tau_E_L_mode, ureg.s), rtol=1e-3)
    np.testing.assert_allclose(magnitude_in_units(tau_E_should_be_H_mode, ureg.s), magnitude_in_units(tau_E_H_mode, ureg.s), rtol=1e-3)
