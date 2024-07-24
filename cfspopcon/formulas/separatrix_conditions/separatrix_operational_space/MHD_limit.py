"""Calculate a function which is used to determine the ideal MHD limit."""

import numpy as np

from ....algorithm_class import Algorithm
from ....unit_handling import Unitfull, convert_units, ureg
from .shared import (
    calc_curvature_drive,
    calc_electron_beta,
    calc_electron_pressure_decay_length_Eich2021H,
    calc_ideal_MHD_wavenumber,
    calc_resistive_ballooning_wavenumber,
    calc_squared_scale_ratio,
)


@Algorithm.register_algorithm(return_keys=["SepOS_MHD_limit"])
def calc_SepOS_ideal_MHD_limit(
    separatrix_electron_density: Unitfull,
    separatrix_electron_temp: Unitfull,
    major_radius: Unitfull,
    magnetic_field_on_axis: Unitfull,
    cylindrical_safety_factor: Unitfull,
    critical_alpha_MHD: Unitfull,
    alpha_t: Unitfull,
    poloidal_sound_larmor_radius: Unitfull,
    ion_to_electron_temp_ratio: float = 1.0,
) -> Unitfull:
    """Calculate a condition function which gives the ideal MHD limit at SepOS_MHD_limit=1.

    If SepOS_MHD_limit < 1, the operating point is stable
    If SepOS_MHD_limit > 1, the transport will increase until SepOS_MHD_limit < 1 (soft limit)

    Equation 12 from :cite:`Eich_2021`

    Args:
        separatrix_electron_density: :term:`glossary link<separatrix_electron_density>`
        separatrix_electron_temp: :term:`glossary link<separatrix_electron_temp>`
        major_radius: :term:`glossary link<major_radius>`
        magnetic_field_on_axis: :term:`glossary link<magnetic_field_on_axis>`
        cylindrical_safety_factor: :term:`glossary link<cylindrical_safety_factor>`
        critical_alpha_MHD: :term:`glossary link<critical_alpha_MHD>`
        alpha_t: :term:`glossary link<alpha_t>`
        poloidal_sound_larmor_radius: :term:`glossary link<poloidal_sound_larmor_radius>`
        ion_to_electron_temp_ratio: :term:`glossary link<ion_to_electron_temp_ratio>`

    Returns:
        :term:`SepOS_MHD_limit`
    """
    k_RBM_factor = np.sqrt(2.0)
    electron_pressure_decay_length = calc_electron_pressure_decay_length_Eich2021H(
        alpha_t=alpha_t, poloidal_sound_larmor_radius=poloidal_sound_larmor_radius
    )

    omega_B = calc_curvature_drive(
        perpendicular_decay_length=electron_pressure_decay_length,
        major_radius=major_radius,
    )
    beta_e = calc_electron_beta(
        electron_density=separatrix_electron_density,
        electron_temp=separatrix_electron_temp,
        magnetic_field_strength=magnetic_field_on_axis,
    )
    epsilon_hat = calc_squared_scale_ratio(
        safety_factor=cylindrical_safety_factor,
        major_radius=major_radius,
        perpendicular_decay_length=electron_pressure_decay_length,
    )

    k_ideal = calc_ideal_MHD_wavenumber(
        beta_e=beta_e,
        epsilon_hat=epsilon_hat,
        omega_B=omega_B,
        tau_i=ion_to_electron_temp_ratio,
        alpha_t=alpha_t,
    )
    k_RBM = (
        calc_resistive_ballooning_wavenumber(
            critical_alpha_MHD=critical_alpha_MHD, alpha_t=alpha_t, omega_B=omega_B
        )
        * k_RBM_factor
    )

    return convert_units(k_ideal / k_RBM, ureg.dimensionless)
