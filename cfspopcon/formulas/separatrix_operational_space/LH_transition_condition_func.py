"""Routine to calculate the LH transition condition function."""
import numpy as np

from ...unit_handling import Unitfull, convert_units, ureg
from .auxillaries import calc_curvature_drive_omega_B, calc_electron_beta, calc_mass_ratio_mu
from .turbulence_wavenumbers import calc_k_EM


def calc_LH_transition_condition(
    electron_density: Unitfull,
    electron_temperature: Unitfull,
    major_radius: Unitfull,
    magnetic_field_strength: Unitfull,
    ion_mass: Unitfull,
    alpha_c: Unitfull,
    alpha_t_turbulence_param: Unitfull,
    lambda_pe: Unitfull,
) -> Unitfull:
    """Calculate LH_transition_condition which gives the LH transition at LH_transition_condition=1.

    If LH_transition_condition < 1, the operating point will be in L-mode
    If LH_transition_condition > 1, the operating point will be in H-mode

    Equation 8 from :cite:`Eich_2021`
    """
    ne = electron_density
    Te = electron_temperature
    B = magnetic_field_strength
    m_i = ion_mass

    beta_e = calc_electron_beta(ne, Te, B)
    mu = calc_mass_ratio_mu(m_i)

    k_EM = calc_k_EM(beta_e=beta_e, mu=mu)
    omega_B = calc_curvature_drive_omega_B(lambda_perp=lambda_pe, major_radius=major_radius)

    flow_shear_stabilisation = alpha_c * k_EM / (1.0 + (alpha_t_turbulence_param * k_EM / alpha_c) ** 2)

    electron_turbulence_destabilisation = 0.5 * alpha_t_turbulence_param
    kinetic_turbulence_destabilisation = k_EM**2 * alpha_t_turbulence_param
    ion_turbulence_destabilisation = alpha_c / (2.0 * k_EM**2) * np.sqrt(omega_B)
    total_destabilisation = electron_turbulence_destabilisation + ion_turbulence_destabilisation + kinetic_turbulence_destabilisation

    return convert_units(flow_shear_stabilisation / total_destabilisation, ureg.dimensionless)
